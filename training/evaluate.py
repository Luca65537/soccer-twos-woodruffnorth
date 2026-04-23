"""Evaluate agents against random opponent. Plays N matches and reports wins."""
import sys
import importlib
import numpy as np
import soccer_twos

N_MATCHES = 10


def evaluate_agent(agent_module_name, n_matches=N_MATCHES):
    env = soccer_twos.make(render=False)
    
    # Load agent
    agent_module = importlib.import_module(agent_module_name)
    agent = agent_module.TeamAgent(env)
    
    wins = 0
    losses = 0
    draws = 0
    
    for match in range(n_matches):
        obs = env.reset()
        team0_reward = 0
        team1_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Our agent controls team 0 (players 0, 1)
            team0_obs = {0: obs[0], 1: obs[1]}
            team0_actions = agent.act(team0_obs)
            
            # Random opponent controls team 1 (players 2, 3)
            team1_actions = {
                2: env.action_space.sample(),
                3: env.action_space.sample(),
            }
            
            actions = {**team0_actions, **team1_actions}
            obs, rewards, dones, infos = env.step(actions)
            
            team0_reward += rewards[0] + rewards[1]
            team1_reward += rewards[2] + rewards[3]
            steps += 1
            
            if any(dones.values()):
                done = True
        
        if team0_reward > team1_reward:
            wins += 1
            result = "WIN"
        elif team0_reward < team1_reward:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        print(f"  Match {match+1}: {result} (team0={team0_reward:.2f}, team1={team1_reward:.2f}, steps={steps})")
    
    env.close()
    print(f"\nResults: {wins}W / {losses}L / {draws}D out of {n_matches} matches")
    return wins, losses, draws


if __name__ == "__main__":
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "agent_reward_shaped"
    print(f"Evaluating {agent_name} vs Random ({N_MATCHES} matches)...")
    evaluate_agent(agent_name)
