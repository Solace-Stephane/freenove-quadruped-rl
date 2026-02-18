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
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
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
  cfg.viewer.distance = 0.8  # smaller robot, closer camera
  cfg.viewer.elevation = -15.0

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
    ".*HAA": 0.1,
    ".*HFE": 0.1,
    ".*KFE": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*HAA": 0.4,
    ".*HFE": 0.4,
    ".*KFE": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.4,
    ".*HFE": 0.4,
    ".*KFE": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["upright"].weight = 1.5  # extra upright reward (light robot)

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].weight = 0.0

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Reduce foot clearance penalty (small robot, low swing is fine).
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].params["target_height"] = (
    0.03  # 30mm (vs 100mm default)
  )
  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.03  # 30mm

  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Slightly penalize large actions to keep motions smooth for real servos.
  cfg.rewards["action_rate_l2"].weight = -0.15

  # -- Terminations --
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # -- Commands: scale down for small robot (max ~0.3 m/s) --
  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.ranges.lin_vel_x = (-0.3, 0.3)
  cmd.ranges.lin_vel_y = (-0.2, 0.2)
  cmd.ranges.ang_vel_z = (-0.4, 0.4)
  cmd.viz.z_offset = 0.15

  # -- Curriculum: disable terrain levels (flat only) --
  cfg.curriculum.pop("terrain_levels", None)

  # -- Curriculum: replace default velocity stages with robot-appropriate ones --
  # The default curriculum starts at lin_vel_x=[-1, 1] which is 3x too fast for
  # this small robot (max ~0.3 m/s). Use gentle ramp-up that stays within the
  # robot's physical capability.
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": [-0.1, 0.1],
        "ang_vel_z": [-0.2, 0.2],
      },
      {
        "step": 60_000,
        "lin_vel_x": [-0.2, 0.2],
        "ang_vel_z": [-0.3, 0.3],
      },
      {
        "step": 150_000,
        "lin_vel_x": [-0.3, 0.3],
        "ang_vel_z": [-0.4, 0.4],
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
