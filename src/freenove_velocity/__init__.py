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
# to clamp scale everywhere — both at construction AND whenever .scale is
# read or .sample()/.log_prob() are called — so no code path can see a
# negative std.
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import torch.distributions as _dist  # noqa: E402

_OriginalNormal = _dist.Normal

_MIN_STD = 1e-6


class _SafeNormal(_OriginalNormal):
    def __init__(self, loc, scale, validate_args=None):
        if isinstance(scale, torch.Tensor):
            scale = scale.clamp(min=_MIN_STD)
        super().__init__(loc, scale, validate_args=validate_args)

    @property
    def scale(self):
        # Always return clamped scale, even if someone mutated it directly
        return self.stddev

    @scale.setter
    def scale(self, value):
        # When rsl_rl assigns distribution.scale = ..., store clamped
        if isinstance(value, torch.Tensor):
            value = value.clamp(min=_MIN_STD)
        self.__dict__["scale"] = value

    @property
    def stddev(self):
        val = self.__dict__.get("scale", super().stddev)
        if isinstance(val, torch.Tensor):
            return val.clamp(min=_MIN_STD)
        return val

    def sample(self, sample_shape=torch.Size()):
        # Belt-and-suspenders: clamp right before sampling
        self.__dict__["scale"] = self.stddev
        return super().sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        self.__dict__["scale"] = self.stddev
        return super().rsample(sample_shape)

    def log_prob(self, value):
        self.__dict__["scale"] = self.stddev
        return super().log_prob(value)


_dist.Normal = _SafeNormal
torch.distributions.Normal = _SafeNormal
print("[freenove_velocity] ✅ Patched torch.distributions.Normal – std clamped ≥ 1e-6 (all paths)")
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
