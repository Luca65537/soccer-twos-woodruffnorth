"""Evaluate student agents vs the pre-trained CEIA baseline agent."""
import sys
import importlib
import numpy as np
import soccer_twos

N_MATCHES = 10


def evaluate(student_module, n_matches=N_MATCHES):
    env = soccer_twos.make(render=False)

    student_mod = importlib.import_module(student_module)
    student = student_mod.TeamAgent(env)

    baseline_mod = importlib.import_module("ceia_baseline_agent")
    baseline = baseline_mod.RayAgent(env)

    wins = losses = draws = 0
    for match in range(n_matches):
        obs = env.reset()
        team0_reward = 0.0
        team1_reward = 0.0
        done = False
        steps = 0
        while not done:
            team0_obs = {0: obs[0], 1: obs[1]}
            team1_obs = {2: obs[2], 3: obs[3]}
            team0_actions = student.act(team0_obs)
            team1_actions = baseline.act(team1_obs)
            actions = {**team0_actions, **team1_actions}
            obs, rewards, dones, infos = env.step(actions)
            team0_reward += rewards[0] + rewards[1]
            team1_reward += rewards[2] + rewards[3]
            steps += 1
            if any(dones.values()):
                done = True
        if team0_reward > team1_reward:
            wins += 1; result = "WIN"
        elif team0_reward < team1_reward:
            losses += 1; result = "LOSS"
        else:
            draws += 1; result = "DRAW"
        print(f"  Match {match+1}: {result} (team0={team0_reward:.2f}, team1={team1_reward:.2f}, steps={steps})", flush=True)

    env.close()
    print(f"\nResults: {wins}W / {losses}L / {draws}D out of {n_matches} matches", flush=True)
    return wins, losses, draws


if __name__ == "__main__":
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "agent_vanilla"
    print(f"Evaluating {agent_name} vs CEIA Baseline ({N_MATCHES} matches)...", flush=True)
    evaluate(agent_name)
