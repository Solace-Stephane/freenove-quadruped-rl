"""Freenove Robot Dog constants for mjlab integration.

Physical robot specs (from Freenove source code):
  - 12 servos: 3 DOF per leg (HAA, HFE, KFE) x 4 legs
  - PCA9685 PWM driver at 50Hz
  - Servo range: 18-162 degrees (mapped to radians here)
  - Link lengths: hip=23mm, femur=55mm, tibia=55mm
  - Body: 136mm x 76mm, standing height ~99mm
  - Total mass: ~450g (estimated)
  - Servos: SG90-class, ~1.8 kg-cm torque (~0.18 Nm)
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

FREENOVE_DOG_XML: Path = _HERE / "xmls" / "freenove_dog.xml"
assert FREENOVE_DOG_XML.exists(), f"MJCF XML not found at {FREENOVE_DOG_XML}"


def get_spec() -> mujoco.MjSpec:
    """Load the Freenove Dog MuJoCo spec from XML."""
    spec = mujoco.MjSpec.from_file(str(FREENOVE_DOG_XML))
    return spec


##
# Actuator config.
##

# SG90-class hobby servo torque limits.
# Typical SG90: 1.8 kg-cm = 0.176 Nm. Use slightly higher for margin.
EFFORT_LIMIT = 0.25  # Nm

# Armature (reflected rotor inertia) - small for hobby servos.
ARMATURE = 0.003

# PD gains: target a 8 Hz natural frequency (hobby servos are slower than
# industrial actuators). Overdamped for stability.
import math

NATURAL_FREQ = 8.0 * 2.0 * math.pi  # 8 Hz in rad/s
DAMPING_RATIO = 2.5  # overdamped for stability

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

FREENOVE_DOG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HAA", ".*HFE", ".*KFE"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

##
# Keyframes / Initial state.
##

# Standing pose: legs straight down with slight bend for stability.
# The robot stands at ~99mm height. With hip=23mm laterally and
# femur+tibia=110mm vertically, we need slight knee bend.
#
# From IK: at default standing (x=0, y=99mm, z=10mm lateral offset),
# HAA ~ 0 rad, HFE ~ 0.15 rad forward lean, KFE ~ -0.3 rad knee bend
#
# Front legs: positive HFE = forward swing, negative KFE = knee bend
# Hind legs: negative HFE = rearward swing (mirrored), positive KFE
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.12),  # base at ~120mm height (99mm + hip offset margin)
    joint_pos={
        ".*HAA": 0.0,  # hips neutral (no abduction)
        "LF_HFE": 0.15,  # front legs slightly forward
        "RF_HFE": 0.15,
        "LH_HFE": -0.15,  # hind legs slightly rearward
        "RH_HFE": -0.15,
        "LF_KFE": -0.3,  # front knees bent
        "RF_KFE": -0.3,
        "LH_KFE": 0.3,  # hind knees bent (mirrored axis)
        "RH_KFE": 0.3,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[LR][FH]_foot$"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision", _foot_regex),
    condim=3,
    priority=1,
    friction=(0.8,),
    solimp={_foot_regex: (0.015, 1, 0.03)},
)

##
# Final config.
##

FREENOVE_DOG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(FREENOVE_DOG_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_freenove_dog_cfg() -> EntityCfg:
    """Get a fresh Freenove Dog robot configuration instance.

    Returns a new EntityCfg each time to avoid mutation issues.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=FREENOVE_DOG_ARTICULATION,
    )


# Precompute action scale from actuator config.
# action_scale = 0.25 * effort_limit / stiffness
# This determines how much the RL action (in [-1, 1]) moves the target position.
FREENOVE_DOG_ACTION_SCALE: dict[str, float] = {}
for _a in FREENOVE_DOG_ARTICULATION.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    _e = _a.effort_limit
    _s = _a.stiffness
    _names = _a.target_names_expr
    assert _e is not None
    for _n in _names:
        FREENOVE_DOG_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_freenove_dog_cfg())
    viewer.launch(robot.spec.compile())
