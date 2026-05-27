# Paper Metadata For HW2 Task D

Task: sonar-based obstacle avoidance for a drone in ROS 2 + Gazebo.

These papers were collected to support processed sonar state design, reward shaping, risk tracking, task-aware local control, and safety filtering.

## A. Zhao et al., 2021

- Filename: `zhao_2021_multisensor_obstacle_avoidance.pdf`
- Full title: Obstacle Avoidance of Multi-Sensor Intelligent Robot Based on Road Sign Detection
- Authors: Jianwei Zhao, Jianhua Fang, Shouzhong Wang, Kun Wang, Chengxiang Liu, Tao Han
- Year: 2021
- Venue/source: Sensors, 21(20), 6777
- DOI/arXiv ID: https://doi.org/10.3390/s21206777
- Source URL: https://www.mdpi.com/1424-8220/21/20/6777
- Downloaded PDF URL: https://pdfs.semanticscholar.org/fe84/49ea4a1999b43c32f1db20e71be4160288c0.pdf
- Status: downloaded. The requested "Behavior Fusion" title was not found as an exact match; the closest exact match is this MDPI Sensors paper by Zhao et al. with the same article number, Sensors 2021, 21, 6777.

Summary: This paper studies obstacle avoidance for a mobile robot that combines ultrasonic distance sensing, camera-based road-sign recognition, and infrared ranging for pit detection. Its central point is that ultrasonic-only avoidance tends to follow fixed obstacle-avoidance routes and cannot easily use task or scene context. The proposed system fuses distance and visual/infrared cues so local obstacle avoidance can respond to both nearby geometry and high-level direction cues. The experiments are on a ground robot, not a drone, but the sensing logic is useful because it treats ultrasonic sensing as one part of a task-aware local controller.

How it affects my design: I should avoid using the sonar reading as a single binary "too close" threshold. Instead, I can turn sonar into a task-aware risk feature that is interpreted together with target direction, velocity, and boundary conditions. This supports a reward term that penalizes close obstacles while still rewarding progress toward the target, because the policy should learn avoidance that remains goal-directed. It also justifies adding explicit safety rules for sonar-invalid or too-close cases rather than expecting the learned policy to infer all low-level danger.

## B. Yuan et al., 2021

- Filename: `yuan_2021_auv_drl_sonar_obstacle_avoidance.pdf`
- Full title: AUV Obstacle Avoidance Planning Based on Deep Reinforcement Learning
- Authors: Jianya Yuan, Hongjian Wang, Honghan Zhang, Changjian Lin, Dan Yu, Chengfeng Li
- Year: 2021
- Venue/source: Journal of Marine Science and Engineering, 9(11), 1166
- DOI/arXiv ID: https://doi.org/10.3390/jmse9111166
- Source URL: https://www.mdpi.com/2077-1312/9/11/1166
- Downloaded PDF URL: https://mdpi-res.com/d_attachment/jmse/jmse-09-01166/article_deploy/jmse-09-01166.pdf
- Status: downloaded.

Summary: This paper applies deep reinforcement learning to AUV obstacle avoidance using processed active-sonar information instead of raw images. The sonar perception model groups many active-sonar beams into lower-dimensional obstacle-distance features, making the observation more compact and learnable. The method compares double-DQN-style reinforcement learning against genetic algorithm and deep-learning baselines in random static, mixed static, and dynamic obstacle environments. The paper is directly relevant to reward shaping because it frames obstacle avoidance as a sequential decision problem with target progress, safety, and collision avoidance objectives.

How it affects my design: This is the strongest reference for representing sonar as processed state. For the drone task, I can use state features such as current pose, velocity, target vector, current sonar range, recent minimum sonar range, and a sonar-risk trend rather than feeding an unprocessed event flag. It also supports using shaped rewards that combine progress-to-goal, distance-to-obstacle penalties, collision penalties, and smooth-control penalties. Even though my action space will likely be continuous PPO velocity commands, the state-design argument carries over cleanly.

## C. Barreto-Cubero et al., 2022

- Filename: `barreto_cubero_2022_sensor_fusion_mobile_robot.pdf`
- Full title: Sensor Data Fusion for a Mobile Robot Using Neural Networks
- Authors: Andres J. Barreto-Cubero, Alfonso Gomez-Espinosa, Jesus Arturo Escobedo Cabello, Enrique Cuan-Urquizo, Sergio R. Cruz-Ramirez
- Year: 2022
- Venue/source: Sensors, 22(1), 305
- DOI/arXiv ID: https://doi.org/10.3390/s22010305
- Source URL: https://www.mdpi.com/1424-8220/22/1/305
- Downloaded PDF URL: https://mdpi-res.com/d_attachment/sensors/sensors-22-00305/article_deploy/sensors-22-00305.pdf
- Status: downloaded.

Summary: This paper builds a neural-network fusion system for a mobile robot using ultrasonic sensors, a RealSense stereo camera, and 2D 360-degree LiDAR. The goal is to create a more reliable distance-to-obstacle estimate and occupancy grid map, especially for obstacles such as glass that can be difficult for one sensor type alone. The system preprocesses sensor data, filters outliers, projects 3D information into a 2D plane, and fuses sensor readings into a more accurate local map. The reported results support the idea that distance sensors should be processed and interpreted in context rather than treated as perfect rays.

How it affects my design: My drone may only use sonar in the assignment, but this paper supports the abstraction of a processed local risk map or sector feature. If I only have one downward sonar topic, I can still borrow the same principle by producing robust features such as clipped range, normalized risk, recent minimum, and validity flags. If I later add more obstacle sensors or virtual range sectors in Gazebo, this paper supports fusing them into a compact observation vector before PPO sees the state. It also gives language for explaining why sonar measurements are noisy, incomplete, and better used as interpreted local proximity cues.

## D. Li et al., 2024

- Filename: `li_2024_auv_state_tracking_sonar_obstacle_avoidance.pdf`
- Full title: An Obstacle Avoidance Strategy for AUV Based on State-Tracking Collision Detection and Improved Artificial Potential Field
- Authors: Yueming Li, Yuhao Ma, Jian Cao, Changyi Yin, Xiangyi Ma
- Year: 2024
- Venue/source: Journal of Marine Science and Engineering, 12(5), 695
- DOI/arXiv ID: https://doi.org/10.3390/jmse12050695
- Source URL: https://www.mdpi.com/2077-1312/12/5/695
- Downloaded PDF URL: https://mdpi-res.com/d_attachment/jmse/jmse-12-00695/article_deploy/jmse-12-00695.pdf
- Status: downloaded.

Summary: This paper proposes a state-tracking collision detection and simulated-annealing potential-field method for AUV obstacle avoidance in dynamic environments. Its main contribution is a proactive collision-risk model that predicts risk between the AUV and moving obstacles, then uses the risk estimate to guide heading and velocity outputs. The paper emphasizes that dynamic obstacle avoidance requires more than instantaneous distance thresholding because future relative motion matters. Although it is a classical planning/control paper rather than an RL paper, it is useful for defining collision-risk features and evaluation metrics.

How it affects my design: I can include risk tracking in the RL observation and reward design, even if the implementation is simple. Useful features include sonar range trend, recent minimum clearance, relative approach risk, and timeout/out-of-bounds indicators. For reward shaping, this paper supports penalizing closing-in danger before collision rather than only applying a terminal crash penalty. It also supports reporting minimum clearance and collision rate, not just final target distance.

## E. Mane et al., 2024/2026 arXiv preprint

- Filename: `mane_2024_eroas_sonar_obstacle_avoidance.pdf`
- Full title: EROAS: 3D Efficient Reactive Obstacle Avoidance System for Autonomous Underwater Vehicles using 2.5D Forward-Looking Sonar
- Authors: Pruthviraj Mane, Allen Jacob George, Rajini Makam, Subhash Gurikar, Rudrashis Majumder, Suresh Sundaram
- Year: 2024 preprint; downloaded version is arXiv v3 dated 2026-05-11
- Venue/source: arXiv preprint; accepted for IEEE Journal of Oceanic Engineering technical communication according to arXiv page
- DOI/arXiv ID: arXiv:2411.05516; https://doi.org/10.48550/arXiv.2411.05516
- Source URL: https://arxiv.org/abs/2411.05516
- Downloaded PDF URL: https://arxiv.org/pdf/2411.05516
- Status: downloaded.

Summary: EROAS is a reactive obstacle-avoidance framework for AUVs using 2D/2.5D forward-looking sonar. It addresses partial observability, limited sonar field of view, and occlusions by combining sonar-profile-guided directional control, short-term obstacle memory, and a spatio-temporal control-barrier-function safety layer. The system is designed to be lightweight and hardware-aware, with simulation and hardware-in-the-loop evaluation. It is highly relevant even though it is not drone-specific, because the core challenge is safe navigation from incomplete acoustic sensing.

How it affects my design: This paper supports adding short-term memory and simple safety filtering around the learned PPO policy. In the drone project, the learned action can be filtered or overridden when sonar indicates dangerously low clearance, invalid readings, or rapidly increasing risk. It also supports including recent sonar history in the observation, because a single sonar sample is partially observable and may not reveal whether risk is increasing. For the report, this is the best reference for explaining why a learned policy should be paired with explicit safety constraints.
