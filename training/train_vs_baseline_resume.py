"""Resume TTBB-long training from the latest checkpoint (no warm-start).
Meant to be chained after train_vs_baseline.py for walltime-backfill friendliness."""
import glob
import os

import ray
from ray import tune
import soccer_twos
from baseline_opponent import VsBaselineEnv, load_baseline

NUM_ENVS_PER_WORKER = 2
ROOT = "/home/hice1/lgianantonio3/scratch/soccertwos/soccer-twos-starter"
BASELINE_WEIGHTS = os.path.join(ROOT, "ceia_baseline_weights.pth")


def find_latest_checkpoint():
    pattern = os.path.join(ROOT, "ray_results", "ttbb_long", "*", "checkpoint_*", "checkpoint-*")
    paths = [p for p in sorted(glob.glob(pattern)) if not p.endswith(".tune_metadata")]
    if not paths:
        raise RuntimeError("No ttbb_long checkpoint found to resume from")
    return paths[-1]


def create_env(env_config):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    job_id = int(os.environ.get("SLURM_JOB_ID", 0))
    env_config.setdefault("base_port", 5005 + (job_id % 500) * 20)
    baseline = load_baseline(BASELINE_WEIGHTS)
    raw = soccer_twos.make(**{k: v for k, v in env_config.items() if k != "num_envs_per_worker"})
    return VsBaselineEnv(raw, baseline)


if __name__ == "__main__":
    ray.init(include_dashboard=False)
    tune.registry.register_env("SoccerVsBaseline", create_env)

    ckpt = find_latest_checkpoint()
    print(f"RESUMING FROM: {ckpt}", flush=True)

    analysis = tune.run(
        "PPO",
        name="ttbb_long",
        restore=ckpt,
        config={
            "num_gpus": 1,
            "num_workers": 2,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerVsBaseline",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {"vf_share_layers": True, "fcnet_hiddens": [512, 512]},
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
        stop={"timesteps_total": 50000000, "time_total_s": 6300},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )
    print("RESUMED RUN DONE")
