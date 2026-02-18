"""Freenove Robot Dog velocity environment configurations.

Follows the same pattern as anymal_c_velocity/env_cfgs.py:
  1. Start from make_velocity_env_cfg() (built-in velocity task defaults)
  2. Set scene entities to our robot
  3. Configure contact sensors for our foot/body geom names
  4. Tune rewards, terminations, and viewer settings
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from freenove_velocity.freenove_dog.freenove_dog_constants import (
  FREENOVE_DOG_ACTION_SCALE,
  get_freenove_dog_cfg,
)


def freenove_dog_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Freenove Dog flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # -- Simulation parameters (tuned for small robot) --
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 50

  # -- Set our robot --
  cfg.scene.entities = {"robot": get_freenove_dog_cfg()}

  # -- Flat terrain (no terrain generator) --
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # -- Remove raycast sensor (no terrain to scan on flat ground) --
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  # -- Contact sensors --
  # Foot geom names match the MJCF model.
  site_names = ("LF", "RF", "LH", "RH")
  geom_names = ("LF_foot", "RF_foot", "LH_foot", "RH_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Only detect base body contact with ground as "illegal".
  # On this tiny 99mm robot, thigh/shank collision geoms naturally
  # brush the ground during walking — that's expected, not illegal.
  # Only the main body box touching ground means the robot has truly fallen.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("base_collision",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # -- Action scaling --
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = FREENOVE_DOG_ACTION_SCALE

  # -- Viewer --
  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 0.6  # smaller robot, closer camera
  cfg.viewer.elevation = -20.0

  # -- Observations: foot height uses our site names --
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # -- Remove height scan observations (flat terrain) --
  if "height_scan" in cfg.observations["actor"].terms:
    del cfg.observations["actor"].terms["height_scan"]
  if "height_scan" in cfg.observations["critic"].terms:
    del cfg.observations["critic"].terms["height_scan"]

  # -- Events: configure for our robot --
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  # -- Rewards: tune for small hobby-servo robot --
  # Pose penalties: wider tolerance since servos are imprecise.
  cfg.rewards["pose"].params["std_standing"] = {
    ".*HAA": 0.15,
    ".*HFE": 0.15,
    ".*KFE": 0.2,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*HAA": 0.5,
    ".*HFE": 0.5,
    ".*KFE": 0.7,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.5,
    ".*HFE": 0.5,
    ".*KFE": 0.7,
  }

  # Halve pose weight — don't let standing-still dominate.
  cfg.rewards["pose"].weight = cfg.rewards["pose"].weight * 0.3

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["upright"].weight = 0.5  # reduced so it doesn't dominate over walking

  # Boost velocity tracking rewards — the main learning signal.
  cfg.rewards["track_linear_velocity"].weight = (
    cfg.rewards["track_linear_velocity"].weight * 3.0
  )
  cfg.rewards["track_angular_velocity"].weight = (
    cfg.rewards["track_angular_velocity"].weight * 2.0
  )

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].weight = 0.0

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Foot clearance: target 15mm lift (achievable for small robot).
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.015
  cfg.rewards["foot_swing_height"].weight = -0.05
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.015

  cfg.rewards["angular_momentum"].weight = 0.0

  # Air time reward: the KEY signal for developing a walking gait.
  # High weight forces the policy to lift feet off the ground.
  cfg.rewards["air_time"].weight = 2.0

  # Light action rate penalty — don't discourage movement.
  cfg.rewards["action_rate_l2"].weight = -0.02

  # -- Terminations --
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # -- Commands: velocity range for visible walking --
  # 0.5 m/s is ~5 body lengths/sec — achievable with hobby servos.
  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.ranges.lin_vel_x = (-0.5, 0.5)
  cmd.ranges.lin_vel_y = (-0.3, 0.3)
  cmd.ranges.ang_vel_z = (-0.8, 0.8)
  cmd.viz.z_offset = 0.15

  # -- Curriculum: disable terrain levels (flat only) --
  cfg.curriculum.pop("terrain_levels", None)

  # -- Curriculum: ramp velocity up over training --
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": [-0.3, 0.3],
        "ang_vel_z": [-0.4, 0.4],
      },
      {
        "step": 100_000,
        "lin_vel_x": [-0.5, 0.5],
        "ang_vel_z": [-0.8, 0.8],
      },
    ]

  # -- Play mode overrides --
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def freenove_dog_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Freenove Dog rough terrain velocity configuration.

  Extends flat config with terrain generation and height scanning.
  Can be used as a second training stage after flat terrain converges.
  """
  cfg = freenove_dog_flat_env_cfg(play=play)

  # Re-enable terrain generator for rough terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"

  # Note: terrain_generator and height_scan would need to be re-added here
  # for rough terrain training. For now, start with flat terrain.

  return cfg
