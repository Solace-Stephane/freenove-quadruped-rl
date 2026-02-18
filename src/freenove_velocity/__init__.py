"""Freenove Dog velocity task registration for mjlab.

Registers two task variants:
  - Mjlab-Velocity-Flat-Freenove-Dog: flat terrain (primary)
  - Mjlab-Velocity-Rough-Freenove-Dog: rough terrain (future)

Includes a monkey-patch for rsl_rl's MlpModel to prevent negative noise std
(RuntimeError: normal expects all elements of std >= 0.0).
"""

# ---------------------------------------------------------------------------
# Monkey-patch: clamp noise std so it never goes negative
# The PyPI rsl_rl doesn't honour noise_std_type="log", so the raw std param
# can drift negative during PPO updates.
#
# We patch torch.normal (the C++ function that actually raises
# "normal expects all elements of std >= 0.0") to clamp std before calling
# the original.  This is the simplest, zero-recursion-risk approach.
# ---------------------------------------------------------------------------
import torch  # noqa: E402

_original_normal = torch.normal


def _safe_normal(mean, std=1.0, *args, **kwargs):
    if isinstance(std, torch.Tensor):
        std = std.clamp(min=1e-6)
    elif isinstance(std, (int, float)) and std < 1e-6:
        std = 1e-6
    return _original_normal(mean, std, *args, **kwargs)


torch.normal = _safe_normal
print("[freenove_velocity] ✅ Patched torch.normal – std clamped ≥ 1e-6")
# ---------------------------------------------------------------------------

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    freenove_dog_flat_env_cfg,
    freenove_dog_rough_env_cfg,
)
from .rl_cfg import freenove_dog_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Freenove-Dog",
    env_cfg=freenove_dog_flat_env_cfg(),
    play_env_cfg=freenove_dog_flat_env_cfg(play=True),
    rl_cfg=freenove_dog_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Freenove-Dog",
    env_cfg=freenove_dog_rough_env_cfg(),
    play_env_cfg=freenove_dog_rough_env_cfg(play=True),
    rl_cfg=freenove_dog_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
