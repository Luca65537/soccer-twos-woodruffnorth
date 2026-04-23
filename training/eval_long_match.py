"""Long-form evaluation that mirrors the Gradescope autograder: play many
sub-episodes across a fixed-duration match, count cumulative goals per team.
Usage:
    python eval_long_match.py <student_agent_module> <match_steps> <n_matches>
"""
import importlib
import sys

import numpy as np
import soccer_twos


def main(student_module, match_steps=3000, n_matches=10, opponent="baseline"):
    env = soccer_twos.make(render=False)

    smod = importlib.import_module(student_module)
    student = smod.TeamAgent(env)

    if opponent == "baseline":
        bmod = importlib.import_module("ceia_baseline_agent")
        opp = bmod.RayAgent(env)
        def opp_act(obs):
            return opp.act({2: obs[2], 3: obs[3]})
    elif opponent == "random":
        def opp_act(obs):
            return {2: env.action_space.sample(), 3: env.action_space.sample()}
    else:
        raise ValueError(opponent)

    overall_wins = overall_losses = overall_draws = 0
    for m in range(n_matches):
        obs = env.reset()
        blue_goals = 0
        orange_goals = 0
        steps = 0
        while steps < match_steps:
            team0_actions = student.act({0: obs[0], 1: obs[1]})
            team1_actions = opp_act(obs)
            actions = {**team0_actions, **team1_actions}
            obs, rewards, dones, infos = env.step(actions)
            # team 0 scoring event: rewards[0] + rewards[1] > 0.5
            # team 1 scoring event: rewards[2] + rewards[3] > 0.5
            if rewards[0] + rewards[1] > 0.5:
                blue_goals += 1
            if rewards[2] + rewards[3] > 0.5:
                orange_goals += 1
            if any(dones.values()):
                obs = env.reset()
            steps += 1
        if blue_goals > orange_goals:
            overall_wins += 1; res = "WIN"
        elif blue_goals < orange_goals:
            overall_losses += 1; res = "LOSS"
        else:
            overall_draws += 1; res = "DRAW"
        print(f"Match {m+1}: {res}  blue(ours)={blue_goals}  orange(opp)={orange_goals}  steps={steps}", flush=True)

    print(f"\nLong-form vs {opponent}: {overall_wins}W / {overall_losses}L / {overall_draws}D over {n_matches} matches of {match_steps} steps", flush=True)
    env.close()


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "agent_vanilla"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    opp = sys.argv[4] if len(sys.argv) > 4 else "baseline"
    main(agent, steps, n, opp)
