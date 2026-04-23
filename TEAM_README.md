# Woodruff North — SoccerTwos MARL Project

Final project for the Deep Reinforcement Learning course.

**Team:** Luca Gianantonio (`lgianantonio3@gatech.edu`) and Tejas Vermani (`tvermani3@gatech.edu`)

## Submitted Agent

`submission/WOODRUFF_NORTH_AGENT.zip` — the zip uploaded to Gradescope.

## Agents Discussed in the Report

| Directory | Role | Final Reward | vs Random | vs Baseline |
|---|---|---|---|---|
| `agent_vanilla/` | Policy-performance (no env mod) | +1.76 | 10W / 0L / 0D | 5W / 5L / 0D |
| `agent_shaped/` | Reward modification | −3.74 | 2W / 3L / 5D | 1W / 9L / 0D |
| `agent_curriculum/` | Curriculum learning (novel concept) | +1.63 | 7W / 3L / 0D | 2W / 8L / 0D |

All three agents share identical model architecture (`512 × 512` tanh MLP, shared trunk, 18-logit MultiDiscrete head for 6 action branches × 3 choices).

## Environment / Training Modifications (rubric 40-pt item)

1. **Reward shaping** — `reward_shaper_single.py`: a Gym wrapper that applies a 10× goal-reward scale and a per-step existential penalty to the base sparse reward. This is the modification for `agent_shaped`.

2. **Curriculum over initial ball position** — `train_curriculum_tvr.py` (`CurriculumCallback` class): three-stage schedule that starts the ball near the opponent goal, advances to midfield, then to anywhere on the pitch. Advancement is reward-triggered (`episode_reward_mean > 0.5`). This is the modification for `agent_curriculum`.

3. **Frozen-baseline opponent env for fine-tuning** — `training/baseline_opponent.py`: standalone PyTorch reimplementation of the CEIA baseline policy and a `VsBaselineEnv` wrapper that puts it on team 1 while exposing a concatenated 672-d team-0 observation to PPO. This enables direct fine-tuning against the exact opponent graded by the autograder.

Training entry points:

- `train_vanilla_tvr.py` — plain PPO, `team_vs_policy` vs random opponent (policy performance).
- `train_shaped_tvr.py` — PPO + `SingleAgentRewardShaper`.
- `train_curriculum_tvr.py` — PPO + `CurriculumCallback`.
- `training/train_vs_baseline.py` — PPO warm-started from `training/vanilla_rllib_weights.pth`, training against the frozen CEIA baseline in long-match episodes (2000-step matches with internal resets, cumulative-goal-diff reward).

## Compute

Training was run on the GT PACE-ICE cluster (NVIDIA L40S GPUs) via SLURM and on Google Colab Pro (A100/H100) for the frozen-baseline fine-tune. Hyperparameters are identical across agents and documented in the report.

## Reproducing on Colab

Open `colab_train_ttbb.ipynb` in Colab, change the runtime to A100 GPU, and run the cells in order. The notebook handles Python 3.8 via `condacolab`, installs a headless X server for Unity, clones this repo, and runs `training/train_vs_baseline.py` under Xvfb.

## Report

PDF in `report/report.pdf` (LaTeX source in `report/report.tex`, CoRL 2026 template).
