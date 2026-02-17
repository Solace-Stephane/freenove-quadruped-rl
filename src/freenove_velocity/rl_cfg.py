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
            entropy_coef=0.01,
        ),
        experiment_name="freenove_dog_velocity",
        max_iterations=5_000,
    )
