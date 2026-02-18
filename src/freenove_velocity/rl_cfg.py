"""RL configuration for Freenove Dog velocity task."""

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
        init_noise_std=1.0,
        noise_std_type="log",
        experiment_name="freenove_dog_velocity",
        max_iterations=8_000,
    )
