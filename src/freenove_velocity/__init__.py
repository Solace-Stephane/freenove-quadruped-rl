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
# can drift negative during PPO updates.  We wrap MlpModel.forward to clamp
# distribution.scale to >= 1e-6 before any .sample() or .log_prob() call.
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from rsl_rl.models.mlp_model import MlpModel  # noqa: E402

_original_mlp_forward = MlpModel.forward


def _safe_forward(self, x, **kwargs):
    result = _original_mlp_forward(self, x, **kwargs)
    if hasattr(self, "distribution") and hasattr(self.distribution, "scale"):
        self.distribution.scale = torch.clamp(self.distribution.scale, min=1e-6)
    return result


MlpModel.forward = _safe_forward
print("[freenove_velocity] ✅ Patched MlpModel.forward – noise std clamped ≥ 1e-6")
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
