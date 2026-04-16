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
TURN_ACTIONS = {"L45", "L22", "R22", "R45"}

FRAME_STACK = 4
OBS_DIM     = 18
STATE_DIM   = OBS_DIM * FRAME_STACK


# ------------------ NETWORK ------------------
class DQN(nn.Module):
    def __init__(self, in_dim=STATE_DIM, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)


# ------------------ FRAME STACK ------------------
class FrameStack:
    def __init__(self, k=FRAME_STACK, obs_dim=OBS_DIM):
        self.k       = k
        self.obs_dim = obs_dim
        self.frames  = deque(maxlen=k)

    def reset(self, obs):
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self._get()

    def step(self, obs):
        self.frames.append(obs.copy())
        return self._get()

    def _get(self):
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


# ------------------ PRIORITIZED REPLAY ------------------
class PrioritizedReplay:
    def __init__(self, cap=200_000, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.cap        = cap
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.buf        = []
        self.priorities = np.zeros(cap, dtype=np.float32)
        self.pos        = 0
        self.size       = 0

    def add(self, s, a, r, s2, done):
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        if self.size < self.cap:
            self.buf.append((s, a, r, s2, done))
        else:
            self.buf[self.pos] = (s, a, r, s2, done)
        self.priorities[self.pos] = max_p
        self.pos  = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, beta):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch, replace=False, p=probs)
        items   = [self.buf[i] for i in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        s  = np.stack([it[0] for it in items]).astype(np.float32)
        a  = np.array([it[1] for it in items], dtype=np.int64)
        r  = np.array([it[2] for it in items], dtype=np.float32)
        s2 = np.stack([it[3] for it in items]).astype(np.float32)
        d  = np.array([it[4] for it in items], dtype=np.float32)

        return s, a, r, s2, d, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def beta_by_step(self, step, total_steps):
        frac = min(step / total_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self):
        return self.size


# ------------------ IMPORT ENV ------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ------------------ BIASED EXPLORATION ------------------
def biased_explore(obs_raw):
    stuck  = int(obs_raw[17])
    signal = int(np.sum(obs_raw[4:12]))

    if stuck:
        return random.choice([ACTION_IDX["L45"], ACTION_IDX["R45"]])
    if signal > 0:
        return ACTION_IDX["FW"]
    if random.random() < 0.70:
        return ACTION_IDX["FW"]
    return random.choice([ACTION_IDX["L45"], ACTION_IDX["L22"],
                          ACTION_IDX["R22"], ACTION_IDX["R45"]])


# ------------------ REWARD ------------------
def compute_reward(s_raw, s2_raw, env_r, action_name, consecutive_turns):
    r = env_r

    prev_signal = int(np.sum(s_raw[4:12]))
    curr_signal = int(np.sum(s2_raw[4:12]))
    attached    = int(s2_raw[16])
    stuck       = int(s2_raw[17])

    if curr_signal > 0 and not stuck:
        r += 0.3 * curr_signal

    if curr_signal > prev_signal:
        r += 2.0

    if curr_signal < prev_signal and prev_signal > 0:
        r -= 1.5

    if attached:
        r += 50.0

    if action_name in TURN_ACTIONS and curr_signal == 0 and not stuck:
        r -= 1.0 * min(consecutive_turns, 10)

    if action_name == "FW" and curr_signal == 0 and not stuck:
        r += 0.2

    return r


# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       required=True)
    ap.add_argument("--out",             default="find_ddqn_per_3.pth")
    ap.add_argument("--episodes",        type=int,   default=5000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=3)
    ap.add_argument("--box_speed",       type=int,   default=3)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=5e-4)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay_cap",      type=int,   default=200_000)
    ap.add_argument("--warmup",          type=int,   default=5000)
    ap.add_argument("--target_sync",     type=int,   default=2000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=2_000_000)
    ap.add_argument("--per_alpha",       type=float, default=0.6)
    ap.add_argument("--per_beta_start",  type=float, default=0.4)
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
    replay = PrioritizedReplay(
        cap=args.replay_cap,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start,
    )
    fstack     = FrameStack()
    steps      = 0
    total_steps = args.episodes * args.max_steps

    def eps_by_episode(ep):
        if ep >= args.episodes:
            return args.eps_end
        frac = ep / args.episodes
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

        raw   = np.array(env.reset(seed=args.seed + ep)).astype(np.float32)
        state = fstack.reset(raw[:OBS_DIM])

        ep_ret            = 0.0
        attached_flag     = False
        consecutive_turns = 0

        for _ in range(args.max_steps):
            eps = eps_by_episode(ep)

            if np.random.rand() < eps:
                a = biased_explore(raw[:OBS_DIM].astype(int))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            action_name = ACTIONS[a]

            if action_name in TURN_ACTIONS:
                consecutive_turns += 1
            else:
                consecutive_turns = 0

            raw2, env_r, done = env.step(action_name, render=False)
            raw2 = np.array(raw2).astype(np.float32)

            r      = compute_reward(
                raw[:OBS_DIM].astype(int),
                raw2[:OBS_DIM].astype(int),
                env_r, action_name, consecutive_turns,
            )
            state2 = fstack.step(raw2[:OBS_DIM])
            ep_ret += r

            attached = int(raw2[16])
            if attached:
                attached_flag = True
                done = True

            replay.add(state, a, r, state2, bool(done))

            raw   = raw2
            state = state2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                beta = replay.beta_by_step(steps, total_steps)
                sb, ab, rb, s2b, db, indices, weights = replay.sample(args.batch, beta)

                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)
                w_t   = torch.tensor(weights)

                with torch.no_grad():
                    next_a   = q(s2b_t).argmax(1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)

                td_errors = (y - pred).detach().cpu().numpy()
                replay.update_priorities(indices, td_errors)

                loss = (w_t * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()

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
            f"attached: {attached_flag} | return: {ep_ret:.1f} | "
            f"eps: {eps_by_episode(ep):.3f} | replay: {len(replay)}"
        )

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
