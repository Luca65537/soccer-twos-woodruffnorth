import os
import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def _get_base_port():
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    return 10000 + (job_id % 1000) * 50


def create_rllib_env_safe(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    if "base_port" not in env_config:
        env_config["base_port"] = _get_base_port()
    env = soccer_twos.make(**env_config)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)
