"""Smoke test: long-match env wrapper."""
import numpy as np
import soccer_twos
from baseline_opponent import VsBaselineEnv, load_baseline

baseline = load_baseline("./ceia_baseline_weights.pth")
raw = soccer_twos.make(render=False)
env = VsBaselineEnv(raw, baseline, match_steps=500)   # short for smoke
print("obs:", env.observation_space.shape, "act:", env.action_space)
obs = env.reset()
total_r = 0.0; steps = 0; done = False
goals_ours = goals_opp = 0
while not done:
    obs, r, done, info = env.step(env.action_space.sample())
    total_r += r
    if r > 0.5: goals_ours += 1
    if r < -0.5: goals_opp += 1
    steps += 1
print(f"steps={steps} total_reward={total_r:.2f} goals_ours={goals_ours} goals_opp={goals_opp}")
assert steps == 500, f"expected 500 steps, got {steps}"
print("SMOKE OK")
env.close()
