# Summary: Champion-Level Drone Racing Using Deep Reinforcement Learning

Source: https://www.nature.com/articles/s41586-023-06419-4

Local PDF: `../pdfs/06_kaufmann_2023_champion_level_drone_racing.pdf`

## Citation

Kaufmann, E., Bauersfeld, L., Loquercio, A., Muller, M., Koltun, V., & Scaramuzza, D. (2023). *Champion-level drone racing using deep reinforcement learning*. Nature, 620, 982-987.

## Main idea

This paper demonstrates that deep reinforcement learning can control a real drone at expert-level racing performance. The task is much harder than this homework, but it is useful evidence that learned control policies can work for aerial robots when the training setup, simulator, and evaluation process are carefully designed.

## Method

The system trains drone racing policies in simulation and addresses the gap between simulation and real-world flight. The policy must handle fast motion, state estimation, and tight trajectories. The work emphasizes that simulator realism, data collection, and robust evaluation matter when transferring learned policies.

## Why it matters for this project

The biggest practical lesson for this homework is not racing speed. It is that timing and dynamics matter. During Part 3 debugging, the environment initially stepped too fast because `spin_once(timeout_sec=0.1)` returned early. That made the training curve misleading. This paper supports the idea that learned drone control depends heavily on matching the training environment to the actual dynamics.

## Design inspiration

- Log training configuration and evaluation configuration.
- Do not trust reward curves alone.
- Use simulation-first training but verify policy behavior visually and with metrics.
- Treat timing and dynamics as part of the RL environment design.

## Connection to Part 3 stages

This paper is relevant to all stages as a high-level example of drone RL. It motivates our decision to:

- fix the control-period timing,
- save run configurations,
- evaluate checkpoints separately,
- and avoid changing training/evaluation targets without recording the change.

## Limitation for our work

The paper uses a much more advanced system and a far larger training setup. This homework is simpler: it controls a Gazebo drone through ROS velocity commands and focuses on staged navigation rather than racing.
