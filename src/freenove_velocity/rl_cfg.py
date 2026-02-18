"""RL configuration for Freenove Dog velocity task.

Uses noise_std_type="log" on the actor so that noise std is parameterised
as exp(param), guaranteeing it stays positive throughout training.
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
            init_noise_std=1.0,
            noise_std_type="log",  # exp(param) → always positive
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            stochastic=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.02,
            learning_rate=0.0003,
            max_grad_norm=0.5,
        ),
        experiment_name="freenove_dog_velocity",
        max_iterations=8_000,
    )
