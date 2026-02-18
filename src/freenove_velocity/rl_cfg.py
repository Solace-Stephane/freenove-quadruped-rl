"""RL configuration for Freenove Dog velocity task.

Matches the ANYmal C reference implementation:
  https://github.com/mujocolab/anymal_c_velocity

Note: noise_std_type="log" was removed because the PyPI version of rsl_rl
(bundled with mjlab) doesn't support it — it only exists in the GitHub dev
version. Using the default "scalar" mode instead, with a safety clamp in
__init__.py to prevent std from going negative during PPO updates.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def freenove_dog_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            stochastic=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.02,
            learning_rate=0.0003,
            max_grad_norm=0.5,
        ),
        experiment_name="freenove_dog_velocity",
        max_iterations=8_000,
    )

