"""Freenove Dog velocity task registration for mjlab.

Registers two task variants:
  - Mjlab-Velocity-Flat-Freenove-Dog: flat terrain (primary)
  - Mjlab-Velocity-Rough-Freenove-Dog: rough terrain (future)

Includes a monkey-patch to prevent negative noise std
(RuntimeError: normal expects all elements of std >= 0.0).
"""

# ---------------------------------------------------------------------------
# Fix: prevent negative std in Normal distributions
#
# The PyPI rsl_rl ignores noise_std_type="log" and stores noise std as a raw
# nn.Parameter (self.std).  During PPO gradient updates this can drift
# negative, crashing torch.distributions.Normal.sample().
#
# Patching torch.normal at the Python level doesn't work — PyTorch calls it
# at the C/CUDA level.  Instead we patch Normal.sample and Normal.rsample
# to clamp scale immediately before the underlying C call.  These are thin
# Python wrappers so we still control them.
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from torch.distributions import Normal  # noqa: E402

_MIN_STD = 1e-6
_orig_rsample = Normal.rsample
_orig_log_prob = Normal.log_prob


def _safe_rsample(self, sample_shape=torch.Size()):
    self.scale = self.scale.clamp(min=_MIN_STD)
    return _orig_rsample(self, sample_shape)


def _safe_log_prob(self, value):
    self.scale = self.scale.clamp(min=_MIN_STD)
    return _orig_log_prob(self, value)


Normal.rsample = _safe_rsample
Normal.log_prob = _safe_log_prob
print("[freenove_velocity] ✅ Patched Normal.rsample/log_prob – scale clamped ≥ 1e-6")
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
