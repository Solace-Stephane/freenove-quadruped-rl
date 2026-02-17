# Freenove Quadruped RL

RL locomotion training for the [Freenove Robot Dog Kit](https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi) using [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + PPO).

## Structure

```
src/freenove_velocity/
  __init__.py                           # Task registration
  env_cfgs.py                           # Environment configs (sensors, rewards, terminations)
  rl_cfg.py                             # RL hyperparameters (PPO)
  freenove_dog/
    freenove_dog_constants.py           # Robot definition (actuators, collision, init state)
    xmls/
      freenove_dog.xml                  # MuJoCo MJCF model
notebooks/
  train.ipynb                           # Google Colab training notebook
deploy/
  deploy.py                             # Sim-to-real deployment for Raspberry Pi
```

## Usage

```bash
# Sanity check: watch the robot stand and fall under zero actions
uv run play Mjlab-Velocity-Flat-Freenove-Dog --agent zero

# Train
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Freenove-Dog \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000

# Deploy to robot
scp deploy/deploy.py policy_checkpoint.pt sxn@192.168.100.234:~/
ssh sxn@192.168.100.234 'python3 deploy.py --checkpoint policy_checkpoint.pt'
```
