#!/usr/bin/env python3
"""Sim-to-real deployment for Freenove Robot Dog.

Runs the trained RL policy on the Raspberry Pi, reading IMU data and
commanding servos via PCA9685 at 50Hz (matching training decimation).

Usage:
  python3 deploy.py --checkpoint policy_checkpoint.pt
  python3 deploy.py --checkpoint policy_checkpoint.pt --dry-run  # no servo output
  python3 deploy.py --checkpoint policy_checkpoint.pt --speed 0.1 --heading 0.0

Prerequisites (on Raspberry Pi):
  pip3 install torch numpy smbus mpu6050-raspberrypi
"""

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants matching the Freenove source code.
# ---------------------------------------------------------------------------

# PCA9685 channel mapping: [leg][joint] -> PWM channel
# Leg order: 0=LF, 1=RF, 2=LH, 3=RH
# Joint order: 0=HAA (hip), 1=HFE (shoulder), 2=KFE (knee)
SERVO_CHANNELS = [
  [4, 3, 2],  # LF: hip=ch4, shoulder=ch3, knee=ch2
  [7, 6, 5],  # RF: hip=ch7, shoulder=ch6, knee=ch5
  [8, 9, 10],  # LH: hip=ch8, shoulder=ch9, knee=ch10
  [11, 12, 13],  # RH: hip=ch11, shoulder=ch12, knee=ch13
]

# Joint ordering in the RL policy output (matches MJCF joint order):
# LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
# LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
JOINT_NAMES = [
  "LF_HAA",
  "LF_HFE",
  "LF_KFE",
  "RF_HAA",
  "RF_HFE",
  "RF_KFE",
  "LH_HAA",
  "LH_HFE",
  "LH_KFE",
  "RH_HAA",
  "RH_HFE",
  "RH_KFE",
]

# Default standing joint positions (radians) - must match INIT_STATE in constants
DEFAULT_JOINT_POS = np.array(
  [
    0.0,
    0.15,
    -0.3,  # LF: HAA, HFE, KFE
    0.0,
    0.15,
    -0.3,  # RF
    0.0,
    -0.15,
    0.3,  # LH
    0.0,
    -0.15,
    0.3,  # RH
  ],
  dtype=np.float32,
)

# Servo angle limits (degrees) from Freenove source
SERVO_MIN_DEG = 18.0
SERVO_MAX_DEG = 162.0

# Policy control frequency (must match training: 200Hz physics / 4 decimation = 50Hz)
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Maximum servo speed: ~60 deg per 100ms for SG90.
# At 50Hz (20ms per step), max ~12 deg per step.
MAX_SERVO_STEP_DEG = 12.0

# Gravity vector (used for projected gravity observation)
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)


def rad_to_servo_deg(
  joint_rad: np.ndarray,
  leg_idx: int,
  joint_idx: int,
) -> float:
  """Convert RL joint angle (radians) to Freenove servo angle (degrees).

  The Freenove code uses a specific mapping from IK angles to servo angles:
    Front legs: shoulder = 90 - (b + cal), knee = c + cal
    Rear legs:  shoulder = 90 + (b + cal), knee = 180 - (c + cal)

  In our MJCF model, joints are defined with consistent axes, so we need
  to map back to the servo convention.
  """
  angle_rad = float(joint_rad)
  angle_deg = math.degrees(angle_rad)

  if joint_idx == 0:  # HAA (hip abduction)
    # HAA: 0 rad = 90 deg servo (centered)
    # Positive rad = outward = increase servo angle for left, decrease for right
    if leg_idx in (0, 2):  # Left legs
      servo_deg = 90.0 + angle_deg
    else:  # Right legs
      servo_deg = 90.0 - angle_deg

  elif joint_idx == 1:  # HFE (shoulder/hip flexion)
    if leg_idx < 2:  # Front legs
      # Front: 0 rad = 90 deg, positive = forward swing = decrease servo
      servo_deg = 90.0 - angle_deg
    else:  # Rear legs
      # Rear: 0 rad = 90 deg, negative = rearward swing = increase servo
      servo_deg = 90.0 + angle_deg

  elif joint_idx == 2:  # KFE (knee flexion)
    if leg_idx < 2:  # Front legs
      # Front: negative KFE = knee bend = positive servo from 0
      servo_deg = -angle_deg  # mapping: 0 rad -> 0 deg, -0.3 rad -> ~17 deg
      servo_deg = max(0, servo_deg)  # knee can't go negative
    else:  # Rear legs
      # Rear: positive KFE = knee bend = 180 - angle
      servo_deg = 180.0 - angle_deg
      servo_deg = min(180, servo_deg)

  else:
    servo_deg = 90.0

  # Clamp to servo range
  return max(SERVO_MIN_DEG, min(SERVO_MAX_DEG, servo_deg))


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
  """Create rotation matrix from Euler angles (ZYX convention)."""
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)

  return np.array(
    [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ],
    dtype=np.float32,
  )


class IMUReader:
  """Read IMU data from MPU-6050 via I2C."""

  def __init__(self):
    try:
      from mpu6050 import mpu6050

      self.sensor = mpu6050(0x68)
      self._available = True
      # Calibration: take baseline readings
      self._calibrate()
    except Exception as e:
      print(f"IMU not available: {e}")
      self._available = False
      self.gyro_bias = np.zeros(3, dtype=np.float32)

  def _calibrate(self, n_samples: int = 100):
    """Compute gyro bias from stationary readings."""
    gyro_sum = np.zeros(3, dtype=np.float32)
    for _ in range(n_samples):
      data = self.sensor.get_gyro_data()
      gyro_sum += np.array([data["x"], data["y"], data["z"]], dtype=np.float32)
      time.sleep(0.005)
    self.gyro_bias = gyro_sum / n_samples
    print(f"IMU calibrated. Gyro bias: {self.gyro_bias}")

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """Read angular velocity (rad/s) and linear acceleration (m/s^2).

    Returns:
        (angular_vel [3], linear_acc [3]) in body frame
    """
    if not self._available:
      return np.zeros(3, dtype=np.float32), np.array([0, 0, 9.81], dtype=np.float32)

    gyro = self.sensor.get_gyro_data()
    accel = self.sensor.get_accel_data()

    ang_vel = np.array(
      [
        math.radians(gyro["x"]) - self.gyro_bias[0],
        math.radians(gyro["y"]) - self.gyro_bias[1],
        math.radians(gyro["z"]) - self.gyro_bias[2],
      ],
      dtype=np.float32,
    )

    lin_acc = np.array(
      [
        accel["x"] * 9.81,
        accel["y"] * 9.81,
        accel["z"] * 9.81,
      ],
      dtype=np.float32,
    )

    return ang_vel, lin_acc


class ServoController:
  """Control PCA9685 servos via I2C."""

  def __init__(self, dry_run: bool = False):
    self.dry_run = dry_run
    self._current_angles = {}  # channel -> current angle in degrees

    if not dry_run:
      try:
        sys.path.insert(
          0,
          str(
            Path.home()
            / "Desktop"
            / "Freenove_Robot_Dog_Kit_for_Raspberry_Pi"
            / "Code"
            / "Server"
          ),
        )
        from PCA9685 import PCA9685
        from Servo import Servo

        self.servo = Servo()
        self._available = True
      except Exception as e:
        print(f"Servo controller not available: {e}")
        self._available = False
    else:
      self._available = False

  def set_angle(self, channel: int, angle_deg: float):
    """Set servo angle with speed limiting."""
    # Speed limit: don't move more than MAX_SERVO_STEP_DEG per step
    if channel in self._current_angles:
      delta = angle_deg - self._current_angles[channel]
      if abs(delta) > MAX_SERVO_STEP_DEG:
        angle_deg = self._current_angles[channel] + math.copysign(
          MAX_SERVO_STEP_DEG, delta
        )

    self._current_angles[channel] = angle_deg

    if self._available and not self.dry_run:
      self.servo.setServoAngle(channel, int(angle_deg))

  def relax(self):
    """Turn off all servo PWM signals (let servos go limp)."""
    if self._available:
      try:
        for ch in range(16):
          self.servo.pwm.setPWM(ch, 0, 0)
      except Exception:
        pass


class PolicyRunner:
  """Load and run the trained RL policy."""

  def __init__(self, checkpoint_path: str):
    import torch

    self.device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(
      checkpoint_path, map_location=self.device, weights_only=False
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Extract actor network from rsl_rl checkpoint format
    # The checkpoint contains 'model_state_dict' with actor/critic params
    if "model_state_dict" in checkpoint:
      state_dict = checkpoint["model_state_dict"]
    elif "actor" in checkpoint:
      state_dict = checkpoint["actor"]
    else:
      state_dict = checkpoint

    # Build actor network matching training architecture: (256, 128, 64)
    # Input: observations, Output: 12 joint position targets
    actor_keys = {k: v for k, v in state_dict.items() if k.startswith("actor")}
    print(f"Actor parameters: {list(actor_keys.keys())}")

    # Determine input size from first layer weights
    first_weight_key = None
    for k in sorted(actor_keys.keys()):
      if "weight" in k:
        first_weight_key = k
        break

    if first_weight_key:
      obs_dim = actor_keys[first_weight_key].shape[1]
      print(f"Observation dimension: {obs_dim}")
    else:
      obs_dim = 48  # fallback estimate

    self.obs_dim = obs_dim
    self.act_dim = 12  # 12 joints

    # Build network
    self.actor = self._build_actor(obs_dim, state_dict)
    self.actor.eval()

    # Running stats for observation normalization (if available)
    self.obs_mean = None
    self.obs_var = None
    if "obs_mean" in checkpoint:
      self.obs_mean = checkpoint["obs_mean"].to(self.device)
      self.obs_var = checkpoint["obs_var"].to(self.device)

  def _build_actor(self, obs_dim: int, state_dict: dict):
    """Build actor network from checkpoint state dict."""
    import torch
    import torch.nn as nn

    # Try to infer architecture from state dict keys
    layers = []
    layer_idx = 0
    prev_dim = obs_dim

    # Look for actor layers
    while True:
      weight_key = None
      bias_key = None

      # Try common naming conventions from rsl_rl
      for prefix in [
        f"actor.{layer_idx}",
        f"actor.layers.{layer_idx}",
        f"actor.mlp.{layer_idx}",
        f"a_net.{layer_idx}",
      ]:
        wk = f"{prefix}.weight"
        bk = f"{prefix}.bias"
        if wk in state_dict:
          weight_key = wk
          bias_key = bk
          break

      if weight_key is None:
        break

      weight = state_dict[weight_key]
      out_dim = weight.shape[0]

      linear = nn.Linear(prev_dim, out_dim)
      linear.weight.data = weight
      if bias_key in state_dict:
        linear.bias.data = state_dict[bias_key]

      layers.append(linear)

      # Add activation (ELU) for hidden layers, not for output
      # We'll check if the next layer exists to decide
      next_exists = any(
        f"actor.{layer_idx + 1}.weight" in state_dict
        or f"actor.layers.{layer_idx + 1}.weight" in state_dict
        or f"actor.mlp.{layer_idx + 1}.weight" in state_dict
        for _ in [None]
      )
      if next_exists:
        layers.append(nn.ELU())

      prev_dim = out_dim
      layer_idx += 1

    if not layers:
      # Fallback: build default (256, 128, 64) architecture
      print("WARNING: Could not infer actor architecture from checkpoint.")
      print("Building default architecture. This may not match the trained model.")
      layers = [
        nn.Linear(obs_dim, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 64),
        nn.ELU(),
        nn.Linear(64, 12),
      ]

    model = nn.Sequential(*layers)
    print(f"Actor network: {model}")
    return model

  def get_action(self, obs: np.ndarray) -> np.ndarray:
    """Run policy inference.

    Args:
        obs: observation vector (matches training observation space)

    Returns:
        12-element array of joint position targets (radians)
    """
    import torch

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    # Normalize observations if stats are available
    if self.obs_mean is not None:
      obs_tensor = (obs_tensor - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

    with torch.no_grad():
      action = self.actor(obs_tensor)

    return action.squeeze(0).numpy()


def build_observation(
  ang_vel: np.ndarray,
  lin_vel: np.ndarray,
  projected_gravity: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  last_action: np.ndarray,
  command: np.ndarray,
) -> np.ndarray:
  """Build the observation vector matching the training observation space.

  Actor observations (from velocity task):
    base_lin_vel:       3  (estimated from IMU integration)
    base_ang_vel:       3  (from gyroscope)
    projected_gravity:  3  (gravity in body frame)
    joint_pos:         12  (relative to default)
    joint_vel:         12  (estimated from position changes)
    actions:           12  (last applied actions)
    command:            3  (vx, vy, wz)
  Total: 48
  """
  # Joint positions relative to default standing pose
  rel_joint_pos = joint_pos - DEFAULT_JOINT_POS

  obs = np.concatenate(
    [
      lin_vel,  # 3: base linear velocity (estimated)
      ang_vel,  # 3: base angular velocity (from gyro)
      projected_gravity,  # 3: gravity vector in body frame
      rel_joint_pos,  # 12: relative joint positions
      joint_vel,  # 12: joint velocities
      last_action,  # 12: last action
      command,  # 3: velocity command [vx, vy, wz]
    ]
  )

  return obs.astype(np.float32)


def main():
  parser = argparse.ArgumentParser(description="Deploy RL policy on Freenove Robot Dog")
  parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)"
  )
  parser.add_argument(
    "--dry-run", action="store_true", help="Run without sending servo commands"
  )
  parser.add_argument(
    "--speed",
    type=float,
    default=0.0,
    help="Forward velocity command (m/s, range: -0.3 to 0.3)",
  )
  parser.add_argument(
    "--lateral",
    type=float,
    default=0.0,
    help="Lateral velocity command (m/s, range: -0.2 to 0.2)",
  )
  parser.add_argument(
    "--heading",
    type=float,
    default=0.0,
    help="Yaw rate command (rad/s, range: -0.4 to 0.4)",
  )
  parser.add_argument(
    "--duration", type=float, default=30.0, help="Run duration in seconds (default: 30)"
  )
  args = parser.parse_args()

  print("=" * 60)
  print("  Freenove Robot Dog - RL Policy Deployment")
  print("=" * 60)
  print(f"  Checkpoint: {args.checkpoint}")
  print(f"  Dry run:    {args.dry_run}")
  print(
    f"  Command:    vx={args.speed:.2f} vy={args.lateral:.2f} wz={args.heading:.2f}"
  )
  print(f"  Duration:   {args.duration:.1f}s")
  print(f"  Frequency:  {CONTROL_HZ:.0f} Hz")
  print("=" * 60)

  # Initialize hardware
  imu = IMUReader()
  servo = ServoController(dry_run=args.dry_run)

  # Load policy
  policy = PolicyRunner(args.checkpoint)

  # State tracking
  joint_pos = DEFAULT_JOINT_POS.copy()
  joint_vel = np.zeros(12, dtype=np.float32)
  last_action = np.zeros(12, dtype=np.float32)
  lin_vel_estimate = np.zeros(3, dtype=np.float32)

  # Velocity command
  command = np.array([args.speed, args.lateral, args.heading], dtype=np.float32)

  # Graceful shutdown
  running = True

  def signal_handler(sig, frame):
    nonlocal running
    print("\nShutting down...")
    running = False

  signal.signal(signal.SIGINT, signal_handler)

  # Move to standing pose first
  print("\nMoving to standing pose...")
  for leg_idx in range(4):
    for joint_idx in range(3):
      channel = SERVO_CHANNELS[leg_idx][joint_idx]
      angle_deg = rad_to_servo_deg(
        DEFAULT_JOINT_POS[leg_idx * 3 + joint_idx], leg_idx, joint_idx
      )
      servo.set_angle(channel, angle_deg)
      if not args.dry_run:
        time.sleep(0.05)
  time.sleep(1.0)

  print("Starting policy execution...\n")

  step = 0
  start_time = time.time()

  try:
    while running and (time.time() - start_time) < args.duration:
      loop_start = time.time()

      # 1. Read IMU
      ang_vel, lin_acc = imu.read()

      # 2. Estimate projected gravity from accelerometer
      # When stationary, accel reads ~[0, 0, 9.81] in body frame
      # Normalize to get gravity direction
      acc_norm = np.linalg.norm(lin_acc)
      if acc_norm > 1.0:
        projected_gravity = lin_acc / acc_norm * 9.81
      else:
        projected_gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

      # 3. Estimate linear velocity (simple integration, drifts over time)
      # In practice, this is very noisy. The policy should be robust to it.
      corrected_acc = lin_acc - projected_gravity
      lin_vel_estimate += corrected_acc * CONTROL_DT
      lin_vel_estimate *= 0.95  # decay to reduce drift

      # 4. Build observation
      obs = build_observation(
        ang_vel=ang_vel,
        lin_vel=lin_vel_estimate,
        projected_gravity=projected_gravity / 9.81,  # normalized
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_action=last_action,
        command=command,
      )

      # 5. Run policy
      action = policy.get_action(obs)

      # 6. Convert action to joint targets
      # Action is a delta from the default pose, scaled by action_scale
      # (action_scale was baked into the training via JointPositionActionCfg)
      joint_targets = DEFAULT_JOINT_POS + action * 0.1  # conservative scaling

      # 7. Update joint velocity estimate
      joint_vel = (joint_targets - joint_pos) / CONTROL_DT

      # 8. Update joint positions
      prev_joint_pos = joint_pos.copy()
      joint_pos = joint_targets.copy()

      # 9. Send to servos
      for leg_idx in range(4):
        for joint_idx in range(3):
          channel = SERVO_CHANNELS[leg_idx][joint_idx]
          angle_deg = rad_to_servo_deg(
            joint_pos[leg_idx * 3 + joint_idx],
            leg_idx,
            joint_idx,
          )
          servo.set_angle(channel, angle_deg)

      last_action = action.copy()

      # 10. Print status periodically
      step += 1
      if step % 50 == 0:
        elapsed = time.time() - start_time
        actual_hz = step / elapsed if elapsed > 0 else 0
        print(
          f"  Step {step:5d} | "
          f"t={elapsed:5.1f}s | "
          f"Hz={actual_hz:4.1f} | "
          f"ang_vel=[{ang_vel[0]:+5.2f},{ang_vel[1]:+5.2f},{ang_vel[2]:+5.2f}] | "
          f"action_norm={np.linalg.norm(action):.3f}"
        )

      # 11. Sleep to maintain control frequency
      elapsed_step = time.time() - loop_start
      sleep_time = CONTROL_DT - elapsed_step
      if sleep_time > 0:
        time.sleep(sleep_time)

  except KeyboardInterrupt:
    pass
  finally:
    # Return to standing pose
    print("\nReturning to standing pose...")
    for leg_idx in range(4):
      for joint_idx in range(3):
        channel = SERVO_CHANNELS[leg_idx][joint_idx]
        angle_deg = rad_to_servo_deg(
          DEFAULT_JOINT_POS[leg_idx * 3 + joint_idx],
          leg_idx,
          joint_idx,
        )
        servo.set_angle(channel, angle_deg)
    time.sleep(1.0)

    elapsed = time.time() - start_time
    print(f"\nDone. Ran {step} steps in {elapsed:.1f}s ({step / elapsed:.1f} Hz)")

    if args.dry_run:
      print("(Dry run - no servo commands were sent)")


if __name__ == "__main__":
  main()
