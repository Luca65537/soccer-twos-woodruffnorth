# WOODRUFF_NORTH_AGENT

**Agent name:** WoodruffNorth-Vanilla-PPO

**Team:** Woodruff North

**Authors:**
- Luca Gianantonio &lt;luca.gianan@gmail.com&gt; (lgianantonio3@gatech.edu)
- Tejas Vermani &lt;tvermani3@gatech.edu&gt;

## Description

PPO policy trained with RLlib / Ray Tune in the `team_vs_policy` (single-agent)
variation of SoccerTwos, with a random opponent on the other team. A single
policy outputs six MultiDiscrete action branches (three per teammate) from the
concatenated 672-dimensional observation of both team members, so one network
controls both players simultaneously.

Network: two fully-connected tanh layers of 512 units, shared value and policy
heads. No reward shaping, no curriculum — the vanilla rubric-baseline agent.

## Training Summary

- Algorithm: PPO (RLlib)
- Variation: `team_vs_policy`, `multiagent=False`
- Environment steps: 15,000,000
- Wall-clock: ~8.5 h on a single NVIDIA L40S (GT PACE-ICE)
- Final mean episode reward: +1.76

## Evaluation

| Opponent | Record |
|---|---|
| Random Agent | 10W / 0L / 0D |
| CEIA Baseline Agent | 5W / 5L / 0D |

## Rubric Mapping

This agent corresponds to the **Policy Performance** criterion of the rubric.
Two additional agents (reward-shaped and curriculum) are discussed in the
final report.
