import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from collections import deque


class GO2Deploy(LeggedRobot):

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,    # cmd     3
            self.projected_gravity,                        # g       3
            self.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            (self.dof_pos - self.default_dof_pos) *
            self.obs_scales.dof_pos,                       # p_t     12
            self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
        ), dim=-1)

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                self.commands[:, :3] * self.commands_scale,    # cmd_t   3
                self.projected_gravity,                        # g_t     3
                self.base_ang_vel * self.obs_scales.ang_vel,   # omega_t 3
                (self.dof_pos - self.default_dof_pos) *
                self.obs_scales.dof_pos,                       # p_t     12
                self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
                self.actions,                                  # a_{t-1} 12
                # add privileged inputs, to foster learning under domain randomization
                self._rand_push_vels[:, :2],                   # 2
                self._added_base_mass,                         # 1
                self._friction_values,                         # 1
                self._base_com_bias,                           # 3
                ctrl_delay,                                    # 1
            ), dim=-1)

        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.obs_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )
        self.critic_history.append(self.privileged_obs_buf)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ------------- Callbacks --------------

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=gs.tc_float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # a_{t-dt}

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost
