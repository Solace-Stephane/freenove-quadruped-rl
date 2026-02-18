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
# can drift negative during PPO updates.  We patch torch.distributions.Normal
# directly so it works regardless of the rsl_rl class names.
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import torch.distributions as _dist  # noqa: E402

_OriginalNormal = _dist.Normal


class _SafeNormal(_OriginalNormal):
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp(scale, min=1e-6)
        super().__init__(loc, scale, validate_args=validate_args)


_dist.Normal = _SafeNormal
torch.distributions.Normal = _SafeNormal
print("[freenove_velocity] ✅ Patched torch.distributions.Normal – std clamped ≥ 1e-6")
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
