"""Freenove Robot Dog velocity environment configurations.

Follows the same pattern as anymal_c_velocity/env_cfgs.py:
  1. Start from make_velocity_env_cfg() (built-in velocity task defaults)
  2. Set scene entities to our robot
  3. Configure contact sensors for our foot/body geom names
  4. Tune rewards, terminations, and viewer settings
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from freenove_velocity.freenove_dog.freenove_dog_constants import (
  FREENOVE_DOG_ACTION_SCALE,
  get_freenove_dog_cfg,
)


# Rough terrain config scaled for the 99mm-tall Freenove robot.
# Feature heights are ~5x smaller than the default ROUGH_TERRAINS_CFG, which
# targets ANYmal/Go1-sized robots (~50cm tall).
FREENOVE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(4.0, 4.0),
  border_width=4.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.3),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.0, 0.02),  # max 2cm steps
      step_width=0.12,
      platform_width=1.0,
      border_width=0.5,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.0, 0.02),
      step_width=0.12,
      platform_width=1.0,
      border_width=0.5,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.35),  # ~20°
      platform_width=1.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.35),
      platform_width=1.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.005, 0.025),  # 5-25mm bumps
      noise_step=0.005,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=0.1,
      amplitude_range=(0.0, 0.03),  # max 3cm wave amplitude
      num_waves=4,
      border_width=0.25,
    ),
  },
  add_lights=True,
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
  # Terminate ONLY on base body touching ground (= truly fallen).
  # NOTE: On this tiny 99mm robot, hip/thigh/shank collision geoms
  # naturally touch the ground even when standing upright. The Go1-style
  # regex pattern (.*_collision) causes 834 terminations/iter here.
  # Anti-crawling is handled by reward shaping instead: air_time=0,
  # strong foot_clearance/swing penalties, and high upright reward.
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
  # Pose: moderate tolerance — tight enough to encourage symmetry,
  # loose enough to allow the range of motion needed for walking.
  cfg.rewards["pose"].params["std_standing"] = {
    ".*HAA": 0.1,
    ".*HFE": 0.1,
    ".*KFE": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*HAA": 0.35,
    ".*HFE": 0.4,
    ".*KFE": 0.55,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.35,
    ".*HFE": 0.4,
    ".*KFE": 0.55,
  }

  # Pose weight: moderate enforcement.
  cfg.rewards["pose"].weight = cfg.rewards["pose"].weight * 0.4

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  # Strong upright reward: the primary anti-crawling mechanism.
  # When crawling, upright ≈ 0.005 (near-zero reward).
  # When standing, upright ≈ 0.98 (near-maximum reward).
  # At weight 2.0, crawling loses ~2.0 reward/step vs walking.
  cfg.rewards["upright"].weight = 2.0

  # Velocity tracking: 2.5× — strong enough to make walking worthwhile,
  # but leaves room for gait quality signals.
  cfg.rewards["track_linear_velocity"].weight = (
    cfg.rewards["track_linear_velocity"].weight * 2.5
  )
  cfg.rewards["track_angular_velocity"].weight = (
    cfg.rewards["track_angular_velocity"].weight * 2.0
  )

  # Body angular velocity: gentle nudge toward symmetric movement.
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].weight = -0.15

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Foot clearance: target 40mm lift (scaled for small robot; default 100mm).
  cfg.rewards["foot_clearance"].params["target_height"] = 0.04
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.04
  # Moderate swing height: encourage all feet to lift, but don't overwhelm.
  cfg.rewards["foot_swing_height"].weight = -0.5
  # Moderate foot slip: discourage dragging without preventing movement.
  cfg.rewards["foot_slip"].weight = -0.2

  cfg.rewards["angular_momentum"].weight = 0.0

  # Air time: DISABLED (weight=0.0) — matches Go1 config.
  # This is the primary exploit vector: robot vibrates legs to farm
  # air_time without walking. Velocity tracking is sufficient.
  cfg.rewards["air_time"].weight = 0.0

  # Action rate: keep default -0.1 — prevents erratic vibrating.

  # -- Terminations --
  # Single illegal_contact covers ALL non-foot geoms (base, hip, thigh, shank).
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
    # Steps are env steps, not iterations. With 4096 envs × 24 steps/iter,
    # each iteration ≈ 98k env steps. Use iter_target × 24 as the step value.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": [-0.3, 0.3],
        "ang_vel_z": [-0.4, 0.4],
      },
      {
        "step": 2000 * 24,     # ~2000 iterations
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


def freenove_dog_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Freenove Dog flat-terrain RUNNING/galloping velocity configuration.

  Trains a separate policy for faster gaits than the walking policy:
    - Velocity command range up to ±1.5 m/s forward (vs ±0.5 walk).
    - Enables air_time reward to encourage a flight phase (trot/gallop).
    - Loosens pose penalties and action-rate penalty to allow snappier motion.
    - Targets higher foot swing height.

  Hardware caveat: SG90 hobby servos cap at ~1 rad/s joint speed, so the
  trained policy may push beyond what the real robot can execute. Sim-only
  demonstrations should work; sim-to-real transfer for fast gaits will be
  limited by actuator bandwidth.
  """
  cfg = freenove_dog_flat_env_cfg(play=play)

  # -- Wider velocity command ranges --
  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.ranges.lin_vel_x = (-1.5, 1.5)
  cmd.ranges.lin_vel_y = (-0.4, 0.4)
  cmd.ranges.ang_vel_z = (-1.2, 1.2)

  # -- Curriculum: ramp velocity in three stages (each iteration ~98k env steps). --
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": [-0.4, 0.4],
        "ang_vel_z": [-0.5, 0.5],
      },
      {
        "step": 1500 * 24,
        "lin_vel_x": [-1.0, 1.0],
        "ang_vel_z": [-0.9, 0.9],
      },
      {
        "step": 3500 * 24,
        "lin_vel_x": [-1.5, 1.5],
        "ang_vel_z": [-1.2, 1.2],
      },
    ]

  # -- Re-enable air time reward to encourage flight phase. --
  # Walking config has this at 0 to avoid the "vibrating legs" exploit; for
  # running we want it positive but small enough that velocity tracking still
  # dominates the reward signal.
  cfg.rewards["air_time"].weight = 0.5

  # -- Higher foot swing target for running gait (60mm vs 40mm walk). --
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  # Reduce penalty weight so the policy is freer to vary swing height.
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_clearance"].weight = -1.0

  # -- Looser pose constraint: running shouldn't look like standing. --
  # std_running is the std applied when the command magnitude is high; widen
  # it considerably so the network isn't penalized for moving its body around.
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.5,
    ".*HFE": 0.6,
    ".*KFE": 0.8,
  }
  cfg.rewards["pose"].weight = cfg.rewards["pose"].weight * 0.6  # was 0.4× → 0.24×

  # -- Reduce action-rate penalty so the policy can move snappier. --
  cfg.rewards["action_rate_l2"].weight = -0.05

  # -- Body angular velocity: relax (some body pitch/roll is natural in run). --
  cfg.rewards["body_ang_vel"].weight = -0.08

  # -- Upright: keep strong but slightly relaxed (running has more body lean). --
  cfg.rewards["upright"].weight = 1.5

  return cfg


def freenove_dog_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Freenove Dog rough terrain velocity configuration.

  Extends the flat config with:
    - procedurally-generated rough terrain scaled for a 99mm robot
    - a downward-facing height-scan raycast sensor mounted on the base
    - height_scan observations in actor + critic
    - terrain_levels curriculum (promote/demote based on tracking success)

  The resulting policy has a different observation size than the flat one
  (adds height-scan rays), so it needs to be trained from scratch.
  """
  cfg = freenove_dog_flat_env_cfg(play=play)

  # -- Enable rough terrain generator scaled for our small robot. --
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = FREENOVE_ROUGH_TERRAINS_CFG
  cfg.scene.terrain.max_init_terrain_level = 2  # start easy

  # -- Add height-scan sensor mounted on the base body. --
  # Grid scaled for the small robot: 0.4m x 0.25m footprint at 5cm resolution
  # gives a 9x6 = 54-ray scan around the base, enough to see step edges
  # without bloating the obs vector.
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.4, 0.25), resolution=0.05),
    max_distance=2.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=False),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  # -- Re-add height_scan observations to actor + critic. --
  # Subtract the nominal stand height (12cm) and clip to ±0.5m, with light
  # noise on the actor side only.
  height_scan_actor = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan", "offset": 0.12},
    noise=Unoise(n_min=-0.02, n_max=0.02),
    clip=(-0.5, 0.5),
  )
  height_scan_critic = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan", "offset": 0.12},
    clip=(-0.5, 0.5),
  )
  cfg.observations["actor"].terms["height_scan"] = height_scan_actor
  cfg.observations["critic"].terms["height_scan"] = height_scan_critic

  # -- Re-enable terrain_levels curriculum (was popped in flat config). --
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_vel,
    params={"command_name": "twist"},
  )

  return cfg
