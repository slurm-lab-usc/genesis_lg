# 🦿 Legged Robotics in Genesis

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)

## Table of Contents

- [🦿 Legged Robotics in Genesis](#-legged-robotics-in-genesis)
  - [Table of Contents](#table-of-contents)
  - [📅 Updates](#-updates)
  - [🌟 Features](#-features)
  - [🧪 Test Results](#-test-results)
  - [🛠 Installation](#-installation)
  - [👋 Usage](#-usage)
    - [🚀 Quick Start](#-quick-start)
    - [📖 Instructions](#-instructions)
  - [🖼️ Gallery](#️-gallery)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [TODO](#todo)

---
## 📅 Updates

<details>
<summary>2025/07/23</summary>

- Added new feature: periodic gait reward for go2_deploy

</details>

<details>
<summary>2025/02/10</summary>

- add measure_heights support, and provide a demo of exteroceptive locomotion ([go2_rough](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_rough))

![](./test_resources//go2_rough_demo.gif)

</details>

<details>
<summary>2024/12/28</summary>

- add [wiki page](https://github.com/lupinjia/genesis_lr/wiki) for instructions

</details>

<details>
<summary>2024/12/26</summary>

- add terrain support, optional terrain type: ["plane", "heightfield"]. 

  ![](./test_resources/terrain_demo.gif)

- move test results to [tests.md](./test_resources/tests.md)

</details>

<details>
<summary>2024/12/24</summary>

- add a new demo environment `bipedal_walker`

</details>

---


## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**

- **Faster and Smaller**
  
  For a go2 walking on the plane task with 4096 envs, the training speed in Genesis is approximately **1.3x** compared to [Isaac Gym](https://developer.nvidia.com/isaac-gym), while the graphics memory usage is roughly **1/2** compared to IsaacGym.

  With this smaller memory usage, it's possible to **run more parallel environments**, which can further improve the training speed.

- Incorporation of various methods in published RL papers
  
  | Method | Paper Link | Location | Materials |
  |--------|------------|----------|-----------|
  | Periodic Gait Reward | [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387) | [`go2_deploy.py`](https://github.com/lupinjia/genesis_lr/blob/main/legged_gym/envs/go2/go2_deploy/go2_deploy.py#L322) | [Comparison between smooth gait function and step gait function](https://github.com/lupinjia/genesis_lr/blob/main/test_resources/gait_reward_type/gait_reward_type.md) |
  | System Identification | [Learning Agile Bipedal Motions on a Quadrupedal Robot](https://arxiv.org/abs/2311.05818) | [go2_sysid](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_sysid) | |


## 🧪 Test Results

For tests conducted on Genesis, please refer to [tests.md](./test_resources/tests.md)

## 🛠 Installation

1. Create a new python virtual env with python>=3.10
2. Install [PyTorch](https://pytorch.org/)
3. Install Genesis following the instructions in the [Genesis repo](https://github.com/Genesis-Embodied-AI/Genesis)
4. Install rsl_rl and tensorboard
   ```bash
   # Install rsl_rl.
   git clone git@github.com:leggedrobotics/rsl_rl.git
   cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517

   # Install tensorboard.
   pip install tensorboard
   ```
5. Install genesis_lr
   ```bash
   git clone git@github.com:lupinjia/genesis_lr.git
   cd genesis_lr
   pip install -e .
   ```

## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helpers.py`), we can run a training session with the following command:

```bash
cd legged_gym/scripts
python train.py --headless # run training without rendering
```

After the training is done, paste the `run_name` under `logs/go2` to `load_run` in `go2_config.py`: 

![](./test_resources/paste_load_run.png)

Then, run `play.py` to visualize the trained model:

![](./test_resources/go2_flat_play.gif)

### 📖 Instructions

For more detailed instructions, please refer to the [wiki page](https://github.com/lupinjia/genesis_lr/wiki)

## 🖼️ Gallery

| Go2 | Bipedal Walker |
|--- | --- |
| ![](./test_resources/go2_flat_play.gif) | ![](./test_resources/bipedal_walker_flat.gif) |

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)

## TODO

- [x] Add domain randomization
- [x] Verify the trained model on real robots.
- [x] Add Heightfield support
- [x] Add meausre_heights support
- [ ] Add go2 deploy demos and instructions (vanilla and explicit estimator)
- [ ] Add teacher-student implementation