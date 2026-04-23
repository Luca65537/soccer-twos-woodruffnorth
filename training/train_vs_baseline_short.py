"""PPO fine-tuning vs frozen CEIA baseline, warm-started from vanilla.

Same 672-d team architecture as vanilla (512-512 tanh shared-trunk FCNet).
Team 1 is driven by the extracted CEIA baseline weights in a PyTorch frozen
forward pass (no Ray-in-Ray). Target: win 9/10 matches vs the baseline.
"""
import os
import socket

import ray
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

import soccer_twos
from baseline_opponent_short import VsBaselineEnv, load_baseline

NUM_ENVS_PER_WORKER = 2
ROOT = "/home/hice1/lgianantonio3/scratch/soccertwos/soccer-twos-starter"
BASELINE_WEIGHTS = os.path.join(ROOT, "ceia_baseline_weights.pth")
VANILLA_WEIGHTS = os.path.join(ROOT, "vanilla_rllib_weights.pth")


def create_env(env_config):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    # SLURM-aware base port to avoid collisions across concurrent jobs
    job_id = int(os.environ.get("SLURM_JOB_ID", 0))
    base_port = 5005 + (job_id % 500) * 20
    env_config.setdefault("base_port", base_port)
    baseline = load_baseline(BASELINE_WEIGHTS)
    raw = soccer_twos.make(**{k: v for k, v in env_config.items() if k != "num_envs_per_worker"})
    return VsBaselineEnv(raw, baseline)


class WarmStartCallback(DefaultCallbacks):
    """Loads vanilla policy weights into the trainable policy on first iter,
    then syncs them to all rollout workers."""

    _loaded = False

    def on_train_result(self, *, trainer=None, result=None, **kwargs):
        if WarmStartCallback._loaded:
            return
        algo = trainer if trainer is not None else kwargs.get("algorithm")
        if algo is None:
            return
        pol = algo.get_policy()
        weights = torch.load(VANILLA_WEIGHTS, map_location="cpu")
        missing, unexpected = pol.model.load_state_dict(weights, strict=False)
        print(f"---- Warm-start: missing={missing}, unexpected={unexpected} ----", flush=True)
        # Broadcast to rollout workers
        new_weights = {"default_policy": pol.get_weights()}
        algo.workers.local_worker().set_weights(new_weights)
        algo.workers.sync_weights()
        WarmStartCallback._loaded = True
        print("---- Warm-start complete: vanilla -> trainable policy synced ----", flush=True)


if __name__ == "__main__":
    ray.init(include_dashboard=False)
    tune.registry.register_env("SoccerVsBaseline", create_env)

    analysis = tune.run(
        "PPO",
        name="ttbb_short",
        config={
            "num_gpus": 1,
            "num_workers": 2,
            "num_envs_per_worker": 2,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": WarmStartCallback,
            "env": "SoccerVsBaseline",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "lr": 3e-4,
            "lambda": 0.95,
            "gamma": 0.998,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "train_batch_size": 8000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            "rollout_fragment_length": 1000,
        },
        stop={
            "timesteps_total": 10000000,
            "time_total_s": 25200,  # 14h, under 16h walltime cap
        },
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )
    best = analysis.get_best_trial("episode_reward_mean", mode="max")
    print("BEST TRIAL:", best)
    print(
        "BEST CKPT:",
        analysis.get_best_checkpoint(trial=best, metric="episode_reward_mean", mode="max"),
    )
    print("TTBB DONE")
