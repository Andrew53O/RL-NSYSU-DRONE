# HW2 Part 3: Clean Six-Stage PPO Curriculum

Part 3 is a clean curriculum pipeline for drone navigation. It avoids the
complex Part 2 reward history and trains skills in order:

1. Vertical altitude control on Gazebo `z`
2. Horizontal movement on Gazebo `x`
3. Combined navigation without sonar
4. Single-obstacle avoidance with sonar
5. Multiple-obstacle avoidance with sonar
6. Sequential targets with sonar obstacle avoidance

Gazebo axes:

```text
x = forward/back
y = left/right
z = altitude
```

Sonar is not used for policy decisions or reward before Stage 4. The observation
still contains sonar slots in Stages 1-3, but they are masked to safe constants
so PPO checkpoints can continue training across all stages.

## Quick Checks

```bash
cd /workspace/HW2_Work/part3
python3 -m py_compile drone_env.py train.py test.py
```

## Smoke Tests

```bash
python3 train.py --stage 1 --variant A --smoke
python3 train.py --stage 1 --variant B --smoke
python3 train.py --stage 2 --variant A --smoke
python3 train.py --stage 2 --variant B --smoke
python3 train.py --stage 3 --variant A --smoke
python3 train.py --stage 3 --variant B --smoke
python3 train.py --stage 4 --smoke
python3 train.py --stage 5 --smoke
python3 train.py --stage 6 --smoke
```

## Training Order

```bash
python3 train.py --stage 1 --variant A --timesteps 30000
python3 train.py --stage 1 --variant B --resume-from models/stage1/variantA/runXXX/best/best_success_model.zip --timesteps 50000
python3 train.py --stage 2 --variant A --resume-from models/stage1/variantB/runXXX/best/best_success_model.zip --timesteps 50000
python3 train.py --stage 2 --variant B --resume-from models/stage2/variantA/runXXX/best/best_success_model.zip --timesteps 70000
python3 train.py --stage 3 --variant A --resume-from models/stage2/variantB/runXXX/best/best_success_model.zip --timesteps 100000
python3 train.py --stage 3 --variant B --resume-from models/stage3/variantA/runXXX/best/best_success_model.zip --timesteps 120000
python3 train.py --stage 4 --resume-from models/stage3/variantB/runXXX/best/best_success_model.zip --timesteps 120000
python3 train.py --stage 5 --resume-from models/stage4/runXXX/best/best_success_model.zip --timesteps 150000
python3 train.py --stage 6 --resume-from models/stage5/runXXX/best/best_success_model.zip --timesteps 200000
```

## Evaluation

```bash
python3 test.py \
  --stage 1 \
  --variant A \
  --model models/stage1/variantA/runXXX/best/best_success_model.zip \
  --episodes 10
```

Each training run saves `run_config.json`; each evaluation saves
`evalXXX.csv` plus `evalXXX_config.json`.
