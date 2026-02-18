"""RL configuration for Freenove Dog velocity task.

Uses noise_std_type="log" (exp parameterization) to guarantee
the noise standard deviation stays positive.
"""

from dataclasses import dataclass, field

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class FreenoveVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: RslRlPpoActorCriticCfg = field(
        default_factory=lambda: RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="log",
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(256, 128, 64),
            activation="elu",
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            entropy_coef=0.02,
            learning_rate=0.0003,
            max_grad_norm=0.5,
        )
    )
    experiment_name: str = "freenove_dog_velocity"
    max_iterations: int = 8_000
