import os

from omegaconf import OmegaConf


def prepare_ray_init_kwargs(ray_init_kwargs=None, default_runtime_env=None, disable_tracking=False):
    """Build ray.init kwargs with optional dashboard and usage stats disabled."""
    ray_init_kwargs = ray_init_kwargs or {}
    default_runtime_env = default_runtime_env or {}

    if disable_tracking:
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_USAGE_STATS_PROMPT_ENABLED", "0")

    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)

    if disable_tracking:
        tracking_runtime_env = {
            "env_vars": {
                "RAY_USAGE_STATS_ENABLED": "0",
                "RAY_USAGE_STATS_PROMPT_ENABLED": "0",
            }
        }
        runtime_env = OmegaConf.merge(tracking_runtime_env, runtime_env)

    prepared_kwargs = {**ray_init_kwargs, "runtime_env": runtime_env}

    if disable_tracking and "include_dashboard" not in prepared_kwargs:
        prepared_kwargs["include_dashboard"] = False

    return OmegaConf.create(prepared_kwargs)
