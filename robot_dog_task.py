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
                mount=self.agent.robot.links[0] if hasattr(self, 'agent') else None,
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
        # Load the agent first
        super()._load_agent(options)
        
        # Get the standing keyframe
        keyframe = self.agent.keyframes["standing"]
        
        # Convert position to numpy array with correct shape and dtype
        position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Create quaternion for identity rotation [w, x, y, z]
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Create the pose with properly formatted numpy arrays
        initial_pose = sapien.Pose(p=position, q=quaternion)
        
        # Set the pose and qpos
        self.agent.robot.set_pose(initial_pose)
        self.agent.robot.set_qpos(keyframe.qpos)

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
            color=[1, 0, 0, 1],
            name="obstacle_cube",
            add_collision=True,
            body_type="static"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            keyframe = self.agent.keyframes["standing"]

            # Convert position to numpy array
            position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # Create pose with properly formatted arrays
            keyframe_pose = sapien.Pose(p=position, q=quaternion)

            self.agent.robot.set_pose(keyframe_pose)
            self.agent.robot.set_qpos(keyframe.qpos)

            # Sample random goals
            xyz = torch.zeros((b, 3), device=self.device)
            noise_scale = 1
            xyz[:, 0] = torch.rand(size=(b,), device=self.device) * noise_scale - noise_scale / 2 + 4
            noise_scale = 2
            xyz[:, 1] = torch.rand(size=(b,), device=self.device) * noise_scale - noise_scale / 2

            # Set goal positions
            for i in range(b):
                goal_position = xyz[i].cpu().numpy().astype(np.float32)
                self.goal.set_pose(sapien.Pose(p=goal_position))

            # Set cube positions for each environment
            robot_pose = self.agent.robot.pose.p

            for i in range(b):
                goal_pose = self.goal.pose.p

                # Extract x-coordinates for distance calculation
                robot_x = robot_pose[i][0]  # Assuming robot_pose is a tensor with shape (b, 3)
                goal_x = goal_pose[0][0]    # Assuming goal_pose is also a tensor with shape (b, 3)

                # Calculate the distance along the x-axis
                robot_x_distance_to_goal = torch.abs(robot_x - goal_x).item()

                # Now we can safely use the scalar value
                cube_forward_delta = max(random.random() * robot_x_distance_to_goal, self.CUBE_HALF_SIZE)
                cube_horizontal_delta = random.random() * self.CUBE_HALF_SIZE

                cube_position = np.array([
                    robot_x + cube_forward_delta,
                    goal_pose[i][1] + cube_horizontal_delta,  # Use the y-coordinate of the goal
                    self.CUBE_HALF_SIZE
                ], dtype=np.float32)

                self.cube.set_pose(sapien.Pose(p=cube_position))

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
        reward[info["fail"]] = -1 # 0
        return reward
    
    def compute_normalized_dense_reward(self, obs, action, info):
        # This can be similar to your existing reward computation logic
        reward = self.compute_dense_reward(obs, action, info)
        
        # Normalize the reward if needed
        max_reward = 10.0  # Define your maximum possible reward
        normalized_reward = reward / max_reward  # Normalize the reward
        
        return normalized_reward


@register_env("AnymalC-Move-v1", max_episode_steps=200)
class AnymalCReachEnv(QuadrupedReachEnv):
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )