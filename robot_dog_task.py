from typing import Any, Dict, List
import numpy as np
import sapien
import torch
from mani_skill.agents.robots.anymal.anymal_c import ANYmalC
from mani_skill.agents.robots.unitree_go.unitree_go2 import UnitreeGo2Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
import random

class QuadrupedReachEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]
    agent: ANYmalC
    default_qpos: torch.Tensor

    _UNDESIRED_CONTACT_LINK_NAMES: List[str] = None

    CUBE_HALF_SIZE = 0.25

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**20),
            scene_config=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.1], target=[1.0, 0, 0.0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links[0],
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-2.0, 1.5, 3], [1.5, 0.0, 0.5])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options)
        self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, floor_width=400)
        self.goal = actors.build_sphere(
            self.scene,
            radius=0.2,
            color=[0, 1, 0, 1],
            name="goal",
            add_collision=False,
            body_type="kinematic",
        )
        
        self.cube = actors.build_cube(
            self.scene,
            half_size=QuadrupedReachEnv.CUBE_HALF_SIZE,
            color=[1, 0, 0, 1], # red
            name="obstacle_cube",
            add_collision=True,
            body_type="static"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            keyframe = self.agent.keyframes["standing"]
            self.agent.robot.set_pose(keyframe.pose)
            self.agent.robot.set_qpos(keyframe.qpos)
            # sample random goal
            xyz = torch.zeros((b, 3))
            noise_scale = 1
            xyz[:, 0] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2 + 4
            noise_scale = 2
            xyz[:, 1] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2
            self.goal.set_pose(Pose.create_from_pq(xyz))

            # randomly place the cube
            robot_pose_p = list(self.agent.robot.pose.p[0])
            goal_pose_p = list(self.goal.pose.p[0])

            robot_x_distance_to_goal = abs(robot_pose_p[0] - goal_pose_p[0])
            
            cube_forward_delta = max(random.random() * robot_x_distance_to_goal, QuadrupedReachEnv.CUBE_HALF_SIZE)
            cube_horizontal_delta = random.random() * QuadrupedReachEnv.CUBE_HALF_SIZE
            
            cube_pose_p = [robot_pose_p[0] + cube_forward_delta, goal_pose_p[1] + cube_horizontal_delta, 0] 
            
            print(f'Robot XYZ: {robot_pose_p}')
            print(f'Goal XYZ: {self.goal.pose.p[0]}')
            print(f'Cube Pose XYZ: {cube_pose_p}')
            self.cube.set_pose(sapien.Pose(p=cube_pose_p))

    def evaluate(self):
        is_fallen = self.agent.is_fallen()
        robot_to_goal_dist = torch.linalg.norm(
            self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )
        reached_goal = robot_to_goal_dist < 0.35
        return {
            "success": reached_goal & ~is_fallen,
            "fail": is_fallen,
            "robot_to_goal_dist": robot_to_goal_dist,
            "reached_goal": reached_goal,
            "is_fallen": is_fallen,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            root_linear_velocity=self.agent.robot.root_linear_velocity,
            root_angular_velocity=self.agent.robot.root_angular_velocity,
            reached_goal=info["success"],
        )
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.goal.pose.p[:, :2],
                robot_to_goal=self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2],
            )
        return obs

    def _compute_undesired_contacts(self, threshold=1.0):
        forces = self.agent.robot.get_net_contact_forces(
            self._UNDESIRED_CONTACT_LINK_NAMES
        )
        contact_exists = torch.norm(forces, dim=-1).max(-1).values > threshold
        return contact_exists

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        robot_to_goal_dist = info["robot_to_goal_dist"]
        reaching_reward = 1 - torch.tanh(1 * robot_to_goal_dist)

        # various penalties:
        lin_vel_z_l2 = torch.square(self.agent.robot.root_linear_velocity[:, 2])
        ang_vel_xy_l2 = (
            torch.square(self.agent.robot.root_angular_velocity[:, :2])
        ).sum(axis=1)
        penalties = (
            lin_vel_z_l2 * -2
            + ang_vel_xy_l2 * -0.05
            + self._compute_undesired_contacts() * -1
            + torch.linalg.norm(self.agent.robot.qpos - self.default_qpos, axis=1)
            * -0.05
        )
        reward = 1 + 2 * reaching_reward + penalties
        reward[info["fail"]] = 0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Override the parent class method to provide a normalized version of the dense reward
        """
        raw_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        max_reward = 3.0  # This is based on the maximum possible reward in compute_dense_reward
        return raw_reward / max_reward

@register_env("AnymalC-Move-v1", max_episode_steps=200)
class AnymalCReachEnv(QuadrupedReachEnv):
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )

# @register_env("UnitreeGo2-Reach-v1", max_episode_steps=200)
# class UnitreeGo2ReachEnv(QuadrupedReachEnv):
#     _UNDESIRED_CONTACT_LINK_NAMES = ["FR_thigh", "RR_thigh", "FL_thigh", "RL_thigh"]

#     def __init__(self, *args, robot_uids="unitree_go2_simplified_locomotion", **kwargs):
#         super().__init__(*args, robot_uids=robot_uids, **kwargs)
#         self.default_qpos = torch.from_numpy(
#             UnitreeGo2Simplified.keyframes["standing"].qpos
#         ).to(self.device)