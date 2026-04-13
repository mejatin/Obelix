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

class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
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

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap=100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch):
        idx   = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s  = np.stack([it.s  for it in items]).astype(np.float32)
        a  = np.array([it.a  for it in items], dtype=np.int64)
        r  = np.array([it.r  for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def biased_explore(obs):
    if obs[17]:
        if obs[5] and not obs[7]:
            return ACTION_IDX["R45"]
        elif obs[7] and not obs[5]:
            return ACTION_IDX["L45"]
        return random.choice([ACTION_IDX["L45"], ACTION_IDX["R45"]])

    if np.sum(obs[4:12]) > 0:
        return ACTION_IDX["FW"]

    if random.random() < 0.70:
        return ACTION_IDX["FW"]
    return random.choice([ACTION_IDX["L45"], ACTION_IDX["L22"],
                          ACTION_IDX["R22"], ACTION_IDX["R45"]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       required=True)
    ap.add_argument("--out",             default="finder_weights.pth")
    ap.add_argument("--episodes",        type=int,   default=5000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay_cap",      type=int,   default=100_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=2000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
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

    def eps_by_ep(ep):
        if ep >= args.episodes:
            return args.eps_end
        frac = ep / args.episodes
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    log_file = open("final.txt", "w")

    for ep in range(args.episodes):
        use_walls = (ep % 2 == 1)
        eps       = eps_by_ep(ep)

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=use_walls,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        s      = np.array(env.reset(seed=args.seed + ep)).astype(np.float32)
        ep_ret = 0.0
        attached_flag = False
        escape_turn   = None

        for _ in range(args.max_steps):

            obs = s.astype(int)
            stuck = int(obs[17])

            if stuck:
                if escape_turn is None:
                    if obs[5] and not obs[7]:
                        escape_turn = ACTION_IDX["R45"]
                    elif obs[7] and not obs[5]:
                        escape_turn = ACTION_IDX["L45"]
                    else:
                        escape_turn = random.choice([ACTION_IDX["L45"], ACTION_IDX["R45"]])
                a = escape_turn
                s2, env_r, done = env.step(ACTIONS[a], render=False)
                s2 = np.array(s2).astype(np.float32)
                if not int(s2[17]):
                    escape_turn = None
                ep_ret += env_r
                s = s2
                steps += 1
                if done:
                    break
                continue

            escape_turn = None

            if np.random.rand() < eps:
                a = biased_explore(obs)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, env_r, done = env.step(ACTIONS[a], render=False)
            s2 = np.array(s2).astype(np.float32)
            ep_ret += env_r

            if int(s2[16]):
                attached_flag = True
                done = True

            replay.add(Transition(s=s, a=a, r=float(env_r), s2=s2, done=bool(done)))
            s = s2
            steps += 1

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

        log_line = (
            f"Ep {ep+1}/{args.episodes} | {wall_str} | "
            f"attached: {attached_flag} | return: {ep_ret:.1f} | "
            f"eps: {eps:.3f} | replay: {len(replay)}"
        )

        print(log_line)
        log_file.write(log_line + "\n")

    log_file.close()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
