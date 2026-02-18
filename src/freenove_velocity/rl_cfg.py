"""RL configuration for Freenove Dog velocity task.

Uses PPO via rsl_rl with a smaller network architecture suited to
the simpler observation/action space of the Freenove robot (12 joints,
no terrain scan on flat ground).
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def freenove_dog_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Freenove Dog velocity task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            stochastic=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.02,
            learning_rate=0.0003,   # 3x lower to prevent noise std going negative
            max_grad_norm=0.5,      # tighter clipping for stability
        ),
        experiment_name="freenove_dog_velocity",
        max_iterations=8_000,
    )
