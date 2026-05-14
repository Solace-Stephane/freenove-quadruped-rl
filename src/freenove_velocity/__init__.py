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
# The PyPI rsl_rl stores noise std as a raw nn.Parameter (self.std) in
# "scalar" mode.  During PPO gradient updates this can drift negative,
# crashing torch.distributions.Normal.sample() at the C/CUDA level.
#
# Strategy: patch MLPModel._update_distribution to clamp self.std BEFORE
# the Normal distribution is created — this is the root fix.  We also keep
# Normal.sample/rsample/log_prob/entropy patches as a safety net.
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from torch.distributions import Normal  # noqa: E402

_MIN_STD = 1e-6


def _sanitize_scale(scale):
    """Replace NaN/inf in scale with _MIN_STD, then clamp to positive."""
    return torch.nan_to_num(scale, nan=_MIN_STD, posinf=1.0, neginf=_MIN_STD).clamp(min=_MIN_STD)


# --- Primary fix: clamp self.std at the source in MLPModel ---
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402

_orig_update_distribution = MLPModel._update_distribution


def _safe_update_distribution(self, obs):
    # Clamp the raw nn.Parameter before it's used to create the distribution
    if hasattr(self, "std") and self.noise_std_type == "scalar":
        self.std.data.copy_(_sanitize_scale(self.std.data))
    _orig_update_distribution(self, obs)


MLPModel._update_distribution = _safe_update_distribution

# --- Safety net: bypass torch.normal entirely to avoid C-level check ---
# torch.normal() has an internal C++ assertion that scale >= 0 which fires
# BEFORE our Python-level clamp when _orig_sample calls it.
# Fix: implement sample/rsample directly using the reparameterization trick.
_orig_log_prob = Normal.log_prob
_orig_entropy = Normal.entropy


def _safe_sample(self, sample_shape=torch.Size()):
    """Sample without calling torch.normal — avoids C-level std check."""
    self.scale = _sanitize_scale(self.scale)
    shape = self._extended_shape(sample_shape)
    with torch.no_grad():
        eps = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device).normal_()
        return self.loc.expand(shape) + eps * self.scale.expand(shape)


def _safe_rsample(self, sample_shape=torch.Size()):
    """Reparameterized sample with sanitized scale."""
    self.scale = _sanitize_scale(self.scale)
    shape = self._extended_shape(sample_shape)
    eps = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device).normal_()
    return self.loc.expand(shape) + eps * self.scale.expand(shape)


def _safe_log_prob(self, value):
    self.scale = _sanitize_scale(self.scale)
    return _orig_log_prob(self, value)


def _safe_entropy(self):
    self.scale = _sanitize_scale(self.scale)
    return _orig_entropy(self)


Normal.sample = _safe_sample
Normal.rsample = _safe_rsample
Normal.log_prob = _safe_log_prob
Normal.entropy = _safe_entropy
print("[freenove_velocity] ✅ Patched MLPModel._update_distribution + Normal – NaN-safe, std clamped ≥ 1e-6")

# ---------------------------------------------------------------------------
# Fix 2: Prevent gradient explosion from corrupting policy weights
#
# Root cause: rsl_rl's adaptive LR schedule ramps the learning rate up to
# 1e-2 when KL divergence is low.  This causes gradient spikes that produce
# NaN/inf losses.  The NaN-safe Normal patches above prevent crashes, but
# the optimizer still applies corrupted gradients, permanently destroying
# the policy.
#
# Solution: monkey-patch PPO.update to:
#   1. Cap adaptive LR upper bound (1e-2 → MAX_ADAPTIVE_LR)
#   2. Check total loss for NaN/inf BEFORE optimizer.step()
#   3. If NaN/inf detected, skip the step and restore weights from snapshot
# ---------------------------------------------------------------------------
import copy
from rsl_rl.algorithms.ppo import PPO  # noqa: E402

_MAX_ADAPTIVE_LR = 5e-4  # Cap: never let adaptive schedule exceed this
_orig_ppo_update = PPO.update
_nan_skip_count = 0


def _safe_ppo_update(self) -> dict[str, float]:
    """PPO update with NaN/inf gradient protection."""
    global _nan_skip_count

    # Snapshot actor + critic weights BEFORE any gradient step
    actor_snapshot = copy.deepcopy(self.actor.state_dict())
    critic_snapshot = copy.deepcopy(self.critic.state_dict())
    optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())

    # Cap the adaptive LR upper bound
    if hasattr(self, 'learning_rate'):
        if self.learning_rate > _MAX_ADAPTIVE_LR:
            self.learning_rate = _MAX_ADAPTIVE_LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    result = _orig_ppo_update(self)

    # Check if any parameter became NaN/inf after the update
    has_nan = False
    for name, param in self.actor.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            has_nan = True
            break
    if not has_nan:
        for name, param in self.critic.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                has_nan = True
                break

    if has_nan:
        _nan_skip_count += 1
        # Restore pre-update weights — the gradient was corrupted
        self.actor.load_state_dict(actor_snapshot)
        self.critic.load_state_dict(critic_snapshot)
        self.optimizer.load_state_dict(optimizer_snapshot)
        # Also reduce LR as a safety measure
        self.learning_rate = max(1e-5, self.learning_rate / 2.0)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        print(
            f"[freenove_velocity] ⚠️  NaN/inf in weights after PPO step — "
            f"ROLLED BACK (skip #{_nan_skip_count}, lr→{self.learning_rate:.2e})"
        )

    # Also cap LR after the update (adaptive schedule runs inside _orig_ppo_update)
    if hasattr(self, 'learning_rate') and self.learning_rate > _MAX_ADAPTIVE_LR:
        self.learning_rate = _MAX_ADAPTIVE_LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    return result


PPO.update = _safe_ppo_update
print(f"[freenove_velocity] ✅ Patched PPO.update – NaN rollback + adaptive LR capped at {_MAX_ADAPTIVE_LR}")
# ---------------------------------------------------------------------------

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    freenove_dog_flat_env_cfg,
    freenove_dog_rough_env_cfg,
    freenove_dog_run_env_cfg,
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

register_mjlab_task(
    task_id="Mjlab-Velocity-Run-Freenove-Dog",
    env_cfg=freenove_dog_run_env_cfg(),
    play_env_cfg=freenove_dog_run_env_cfg(play=True),
    rl_cfg=freenove_dog_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
