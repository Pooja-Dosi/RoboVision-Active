# RoboVision – Learning Hand-Eye Coordination in Robotics

This repository contains the code for my MSc Computing Science dissertation at the University of Glasgow (2025).

The project explores how active vision can improve robotic manipulation using a combination of **Behavioural Cloning (BC)** and **Reinforcement Learning (RL)**.

The main idea was to investigate whether allowing a robot to actively move its camera and track an object would help it perform a pick-and-place task more reliably than using a fixed camera.

## About the project

I developed a simulated robotic environment in **MuJoCo**, using a 7-DoF Franka Panda robotic arm with an additional pan-tilt camera.

The task was fairly simple: the robot had to locate a ball, pick it up and place it inside a bucket. However, I wanted to explore how the robot's visual attention affected its ability to complete the task.

I used two learning approaches:

- **Behavioural Cloning (BC)** to train the robotic arm using expert demonstrations.
- **Proximal Policy Optimization (PPO)** to train the camera to keep the target object in view.

The two policies were then combined into a perception-action loop, where the camera adjusts its view while the arm performs the manipulation task.

I also compared the system against a static-camera setup to see whether active vision actually made a difference.

## Tools and libraries

The project was developed in Python, mainly using:

- **MuJoCo** – for the robotic simulation environment
- **PyTorch** – for training the Behavioural Cloning model
- **ResNet-18** – as the visual feature extractor for the hand policy
- **Stable-Baselines3** – for PPO training
- **Gymnasium** – for the reinforcement learning environment
- **NumPy** – for processing and storing demonstration data

## Implementation

### Behavioural Cloning

I first generated expert demonstrations for the pick-and-place task using scripted robot movements.

These demonstrations were used to train a ResNet-18-based policy that takes camera images as input and predicts the robot's joint actions and gripper state.

I also experimented with inverse kinematics (IK) assistance to compare purely learned control with a combination of learned and scripted movements.

### Reinforcement Learning for the camera

For the active camera, I used PPO to train a policy that could move the camera horizontally and vertically to keep the ball centred in its view.

The camera receives RGB observations and selects from five possible actions: stay, pan left, pan right, tilt up or tilt down.

The reward function encourages the camera to keep the target near the centre of the image.

### Combining the two policies

Once both policies were trained, I integrated them into the same simulation.

The camera policy controls where the robot looks, while the Behavioural Cloning policy controls the arm.

The aim was to see whether this coordination could improve the robot's ability to locate, grasp and place the ball.

## Results

I ran several experiments to evaluate the system, including comparisons between active and static cameras, and between different hand-control configurations.

One of the main findings was that the active camera improved the robot's success rate.

| Configuration | Task success rate |
|---|---|
| BC with active camera | 70% |
| BC with static camera | 40% |
| BC–RL with active camera | 100% |
| BC–RL with static camera | 60% |

These configurations were each evaluated over 10 trials.

I also compared the integrated BC–RL system with the eye controller enabled and disabled. The eye-on configuration completed all 10 trials successfully, while the eye-off configuration completed 7 out of 10.

In a separate experiment using 50 randomized starting positions, the robot successfully completed 36 trials (72%).

Although the results were encouraging, the system still had limitations, particularly around generalisation to different object positions and the reliance on a simulated environment.

## Project structure

Some of the main files in this repository are:

| File | Description |
|---|---|
| `eye_env.py` | MuJoCo environment and eye-control setup |
| `run.py` | Main simulation script |
| `train_bc_resnet.py` | Training the Behavioural Cloning policy |
| `train_eye_ppo.py` | PPO training for the camera |
| `train_eye_ppo_center.py` | Gaze-centering training |
| `continue_eye_ppo.py` | Continuing PPO training from checkpoints |
| `eval_bc_batch.py` | Batch evaluation of the BC policy |
| `eval_bc_viewer.py` | Visual evaluation of the hand policy |
| `eval_eye_ppo_viewer.py` | Visual evaluation of the eye policy |
| `clean_dataset.py` | Cleaning demonstration data |
| `panda_assets/` | Franka Panda robot model and assets |
| `demos/` | Demonstration data |
| `scripts/` | Additional training and evaluation utilities |

Large model checkpoints and training outputs have been excluded from the repository.

## Limitations

There are a few things I would improve if I were to extend this work.

The system was tested only in simulation and focused on a single pick-and-place task. It would be interesting to see how well the approach works with different objects, more complex manipulation tasks or eventually a physical robot.

Another challenge was getting the eye and hand policies to work together consistently. Although the combined approach performed well in the experiments, some configurations still required additional controller assistance.

## Demo

[Watch the RoboVision simulation demo on YouTube](https://youtu.be/4POp6GB4mPE)


## Reference

This project was inspired by the research paper *Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop* (Kerr et al., 2025).

---

**Pooja Dosi**  
MSc Computing Science  
University of Glasgow, 2025
