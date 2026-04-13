"""
Pusher Trainer — Double DQN.
Trains the agent to push the attached box to the boundary.
Spawns near box every episode for dense push experience.

State: obs[:18] + attachment_side = 19 features
No frame stacking needed — box is always visible once attached.
"""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ------------------ NETWORK ------------------
class DQN(nn.Module):
    def __init__(self, in_dim=19, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return self.net(x)


# ------------------ REPLAY ------------------
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap=50_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch):
        items = random.sample(self.buf, batch)
        s  = np.stack([it.s  for it in items]).astype(np.float32)
        a  = np.array([it.a  for it in items], dtype=np.int64)
        r  = np.array([it.r  for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# ------------------ STATE ------------------
def get_attachment_side(obs):
    if not obs[16]: return 0.0
    if obs[5] and not obs[7]: return -1.0
    if obs[7] and not obs[5]: return 1.0
    return 0.0


def build_state(obs, side):
    return np.append(obs[:18].astype(np.float32), np.float32(side))


# ------------------ REWARD ------------------
def compute_reward(s, s2, env_r, env):
    r = env_r

    attached_now  = int(s2[16])
    attached_prev = int(s[16])
    stuck_now     = int(s2[17])

    dx    = env.box_center_x - env.box_center_x_prev
    dy    = env.box_center_y - env.box_center_y_prev
    moved = abs(dx) + abs(dy)

    if attached_now and moved > 0:
        r += 3.0

    if attached_prev and not attached_now:
        r -= 5.0

    if stuck_now:
        r -= 2.0

    return r


# ------------------ IMPORT ENV ------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       required=True)
    ap.add_argument("--out",             default="pusher_weights.pth")
    ap.add_argument("--episodes",        type=int,   default=1000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=0)   # static box — once attached no blinking
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay_cap",      type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=1000)
    ap.add_argument("--target_sync",     type=int,   default=500)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=200_000)
    ap.add_argument("--seed",            type=int,   default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DQN()
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay_cap)
    steps  = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for ep in range(args.episodes):

        use_walls = (ep % 2 == 1)

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=use_walls,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        env.reset(seed=args.seed + ep)

        # spawn near box — every episode starts in push context
        raw = None
        for _ in range(20):
            angle = random.uniform(0, 2 * np.pi)
            dist  = random.uniform(1.2, 1.8) * env.bot_radius

            env.bot_center_x = int(env.box_center_x + dist * np.cos(angle))
            env.bot_center_y = int(env.box_center_y + dist * np.sin(angle))

            dx = env.box_center_x - env.bot_center_x
            dy = env.box_center_y - env.bot_center_y
            env.facing_angle  = np.degrees(np.arctan2(dy, dx))
            env.facing_angle += random.uniform(-10, 10)

            env._update_frames(show=False)
            env.get_feedback()
            raw = np.array(env.sensor_feedback).astype(int)

            A, R = env.arena_size, env.bot_radius
            if raw[17] == 0 and (R+20 < env.bot_center_x < A-R-20) and (R+20 < env.bot_center_y < A-R-20):
                break

        side  = get_attachment_side(raw)
        state = build_state(raw, side)
        ep_ret = 0.0

        for _ in range(args.max_steps):

            env.box_center_x_prev = env.box_center_x
            env.box_center_y_prev = env.box_center_y

            # unwedge
            if raw[17]:
                if raw[5] and not raw[7]:
                    a = ACTION_IDX["R45"]
                elif raw[7] and not raw[5]:
                    a = ACTION_IDX["L45"]
                else:
                    a = random.choice([ACTION_IDX["L45"], ACTION_IDX["R45"]])
                raw2, _, done = env.step(ACTIONS[a], render=False)
                raw = np.array(raw2).astype(int)
                if done:
                    break
                continue

            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            raw2, env_r, done = env.step(ACTIONS[a], render=False)
            raw2 = np.array(raw2).astype(int)

            new_side = get_attachment_side(raw2)
            if new_side != 0.0:
                side = new_side

            r      = compute_reward(raw, raw2, env_r, env)
            state2 = build_state(raw2, side)
            ep_ret += r

            replay.add(Transition(s=state, a=a, r=r, s2=state2, done=bool(done)))

            raw   = raw2
            state = state2
            steps += 1

            # ---- DDQN UPDATE ----
            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)

                with torch.no_grad():
                    next_a   = q(s2b_t).argmax(1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        wall_str = "wall" if use_walls else "no wall"
        print(
            f"Ep {ep+1}/{args.episodes} | {wall_str} | "
            f"return: {ep_ret:.1f} | eps: {eps_by_step(steps):.3f} | "
            f"replay: {len(replay)}"
        )

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()