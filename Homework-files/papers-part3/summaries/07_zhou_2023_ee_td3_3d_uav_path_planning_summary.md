# Summary: Three-Dimensional Path Planning of UAVs in a Complex Dynamic Environment Based on EE-TD3

Source: https://www.mdpi.com/2073-8994/15/7/1371

Local PDF: `../pdfs/07_zhou_2023_ee_td3_uav_3d_path_planning.pdf`

## Citation

Zhang, D., Li, X., Ren, G., Yao, J., Chen, K., & Li, X. (2023). *Three-Dimensional Path Planning of UAVs in a Complex Dynamic Environment Based on Environment Exploration Twin Delayed Deep Deterministic Policy Gradient*. Symmetry, 15(7), 1371.

## Main idea

This paper proposes an Environment Exploration TD3 method for 3D UAV path planning. It emphasizes that UAV path planning in a 3D environment is harder than simple 2D planning because the agent must reason about altitude, horizontal motion, and obstacle constraints together.

## Method

The method builds on TD3 and adds an environment-exploration coding mechanism. The paper also uses dynamic reward design to guide the agent and reduce sparse-reward difficulty. It compares DDPG, TD3, and the proposed EE-TD3 method and reports better convergence and path-planning performance from the improved approach.

## Why it matters for this project

This paper supports the idea that reward design is central to UAV RL. In our Part 3 experiments, a single distance reward was not enough. We needed axis-specific progress and stability penalties to make Stage 1 altitude control work reliably. For future Stage 2 and Stage 3 training, this paper supports continuing to use x/y/z-aware reward terms.

## Design inspiration

- Treat UAV navigation as a 3D continuous-control problem.
- Use dynamic reward shaping instead of only terminal reward.
- Add axis-aware progress terms to improve learning.
- Compare training reward with task success, because convergence is not only about the curve looking smooth.

## Connection to Part 3 stages

This paper is especially relevant to Stage 3, where vertical and horizontal movement are combined. It also supports later obstacle stages because 3D path planning must balance target progress with safety constraints.

## Limitation for our work

The paper uses TD3 and a more complex environment-exploration design. This homework keeps PPO and a simpler observation space to stay within the deadline.
