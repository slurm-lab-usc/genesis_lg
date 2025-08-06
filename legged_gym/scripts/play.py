import genesis as gs
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 5)
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    for i in range(2):
        env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
        env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4
    env_cfg.env.debug_viz = True
    if RECORD_FRAMES or FOLLOW_ROBOT:
        env_cfg.viewer.add_camera = True  # use a extra camera for moving
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 5
    env_cfg.noise.add_noise = True
    env_cfg.asset.fix_base_link = False
    # initial state randomization
    env_cfg.init_state.yaw_angle_range = [0., 0.]
    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-1., 1.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.commands.ranges.heading = [0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    # for MOVE_CAMERA
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # for FOLLOW_ROBOT
    camera_lookat_follow = np.array(env_cfg.viewer.lookat)
    camera_deviation_follow = np.array([0., 3., -1.])
    camera_position_follow = camera_lookat_follow - camera_deviation_follow
    # for RECORD_FRAMES
    stop_record = 400
    if RECORD_FRAMES:
        env.floating_camera.start_recording()

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
            env.floating_camera.render()
        if FOLLOW_ROBOT:
            # refresh where camera looks at(robot 0 base)
            camera_lookat_follow = env.base_pos[robot_index, :].cpu().numpy()
            # refresh camera's position
            camera_position_follow = camera_lookat_follow - camera_deviation_follow
            env.set_camera(camera_position_follow, camera_lookat_follow)
            env.floating_camera.render()
        if RECORD_FRAMES and i == stop_record:
            env.floating_camera.stop_recording(save_to_filename="go2_rough_demo.mp4", fps=30)
            print("Saved recording to " + "go2_rough_demo.mp4")
        
        # print debug info
        # print("base lin vel: ", env.base_lin_vel[robot_index, :].cpu().numpy())
        # print("base yaw angle: ", env.base_euler[robot_index, 2].item())
        # print("base height: ", env.base_pos[robot_index, 2].cpu().numpy())
        # print("foot_height: ", env.link_pos[robot_index, env.feet_indices, 2].cpu().numpy())
        
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.link_contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    play(args)
