from typing import Any, Dict, List

import numpy as np
import sapien
import torch
import random 
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
import math

class QuadrupedReachEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]
    agent: ANYmalC
    default_qpos: torch.Tensor

    _UNDESIRED_CONTACT_LINK_NAMES: List[str] = None

    CUBE_HALF_SIZE = 0.25
    CUBE_HEIGHT = 1

    GOAL_DISTANCE = 5

    MIN_DISTANCE_BETWEEN_CUBES = 1.5

    MAX_CUBE_HORIZONTAL_DEVIATION = 2

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
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
                # mount=self.agent.robot.links[0],
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options)

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

        self.cube_1 = actors.build_box(
            self.scene,
            half_sizes=[QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HEIGHT],
            color=[1, 0, 0, 1],
            name="obstacle_box_1",
            add_collision=True,
            body_type="kinematic"
        )

        self.cube_2 = actors.build_box(
            self.scene,
            half_sizes=[QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HEIGHT],
            color=[1, 0, 0, 1],
            name="obstacle_box_2",
            add_collision=True,
            body_type="kinematic"
        )

        self.cube_3 = actors.build_box(
            self.scene,
            half_sizes=[QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HEIGHT],
            color=[1, 0, 0, 1],
            name="obstacle_box_3",
            add_collision=True,
            body_type="kinematic"
        )

        self.cube_4 = actors.build_box(
            self.scene,
            half_sizes=[QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HEIGHT],
            color=[1, 0, 0, 1],
            name="obstacle_box_4",
            add_collision=True,
            body_type="kinematic"
        )

        self.cube_5 = actors.build_box(
            self.scene,
            half_sizes=[QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HALF_SIZE, QuadrupedReachEnv.CUBE_HEIGHT],
            color=[1, 0, 0, 1],
            name="obstacle_box_5",
            add_collision=True,
            body_type="kinematic"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            keyframe = self.agent.keyframes["standing"]
            self.agent.robot.set_pose(keyframe.pose)
            self.agent.robot.set_qpos(keyframe.qpos)
            # sample random goal
            xyz = torch.tensor([QuadrupedReachEnv.GOAL_DISTANCE, 0, 0]).repeat(b, 1)
            self.goal.set_pose(Pose.create_from_pq(xyz))

            robot_pose_p = list(self.agent.robot.pose.p[0])
            goal_pose_p = list(self.goal.pose.p[0])
            robot_x_distance_to_goal = abs(robot_pose_p[0] - goal_pose_p[0])

            random_cube_positions = QuadrupedReachEnv.generate_random_cube_positions(
                robot_x_distance_to_goal=robot_x_distance_to_goal
            )

            cube_1_pose_p = np.array([
                float(robot_pose_p[0] + random_cube_positions[0][0]),
                float(goal_pose_p[1] + random_cube_positions[0][1]),
                0.0
            ], dtype=np.float32)

            cube_2_pose_p = np.array([
                float(robot_pose_p[0] + random_cube_positions[1][0]),
                float(goal_pose_p[1] + random_cube_positions[1][1]),
                0.0
            ], dtype=np.float32)

            cube_3_pose_p = np.array([
                float(robot_pose_p[0] + random_cube_positions[2][0]),
                float(goal_pose_p[1] + random_cube_positions[2][1]),
                0.0
            ], dtype=np.float32)

            cube_4_pose_p = np.array([
                float(robot_pose_p[0] + random_cube_positions[3][0]),
                float(goal_pose_p[1] + random_cube_positions[3][1]),
                0.0
            ], dtype=np.float32)

            cube_5_pose_p = np.array([
                float(robot_pose_p[0] + random_cube_positions[4][0]),
                float(goal_pose_p[1] + random_cube_positions[4][1]),
                0.0
            ], dtype=np.float32)

            self.cube_1.set_pose(sapien.Pose(p=cube_1_pose_p))
            self.cube_2.set_pose(sapien.Pose(p=cube_2_pose_p))
            self.cube_3.set_pose(sapien.Pose(p=cube_3_pose_p))
            self.cube_4.set_pose(sapien.Pose(p=cube_4_pose_p))
            self.cube_5.set_pose(sapien.Pose(p=cube_5_pose_p))

    def evaluate(self):
        is_fallen = self.agent.is_fallen()
        robot_to_goal_dist = torch.linalg.norm(
            self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )

        robot_to_cube_1_dist = torch.linalg.norm(
            self.cube_1.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )

        robot_to_cube_2_dist = torch.linalg.norm(
            self.cube_2.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )

        robot_to_cube_3_dist = torch.linalg.norm(
            self.cube_3.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )

        reached_goal = robot_to_goal_dist < 0.35
        return {
            "success": reached_goal & ~is_fallen,
            "fail": is_fallen,
            "robot_to_goal_dist": robot_to_goal_dist,
            "robot_to_cube_1_dist": robot_to_cube_1_dist,
            "robot_to_cube_2_dist": robot_to_cube_2_dist,
            "robot_to_cube_3_dist": robot_to_cube_3_dist,
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
        def compute_obstacle_penalties_and_clearance_reward_sum(proximity_penalty_steepness, proximity_penalty_strength, safety_radius_coefficient):
            # robot_to_cube_1_dist = info["robot_to_cube_1_dist"]
            obstacle_penalty_sum = 0
            clearance_reward_sum = 0
            robot_to_cube_distances = [info[f"robot_to_cube_{i}_dist"] for i in range(1, 4)]
            for robot_to_cube_dist in robot_to_cube_distances:
                safety_radius = safety_radius_coefficient * QuadrupedReachEnv.CUBE_HALF_SIZE  # Set safety distance (e.g., twice the obstacle size)
                proximity_penalty = torch.where(
                    robot_to_cube_dist < safety_radius,
                    -proximity_penalty_strength * (1 - torch.tanh(proximity_penalty_steepness * (safety_radius - robot_to_cube_dist))),
                    torch.zeros_like(robot_to_cube_dist),
                )
                # Collision penalty: Strong penalty for touching the obstacle
                corner_zone_threshold = QuadrupedReachEnv.CUBE_HALF_SIZE * 1.1  # Slightly larger than the cube
                collision_penalty = torch.where(
                    robot_to_cube_dist < corner_zone_threshold,
                    torch.full_like(robot_to_cube_dist, -7),  # Stronger penalty near corners
                    torch.zeros_like(robot_to_cube_dist),
                )

                # Clearance reward: Encourage maintaining a safe margin
                safe_margin = safety_radius + 0.2  # Additional clearance
                clearance_reward = torch.where(
                    robot_to_cube_dist > safe_margin,
                    0.5 * (robot_to_cube_dist - safe_margin),  # Reward for extra clearance
                    torch.zeros_like(robot_to_cube_dist),
                )

                # Combine penalties and rewards
                obstacle_penalty = proximity_penalty + collision_penalty
                obstacle_penalty_sum += obstacle_penalty
                clearance_reward_sum += clearance_reward
            return obstacle_penalty_sum, clearance_reward_sum
            
        robot_to_goal_dist = info["robot_to_goal_dist"]
        reaching_reward = 1 - torch.tanh(1 * robot_to_goal_dist)

        obstacle_penalty_sum, clearance_reward_sum = compute_obstacle_penalties_and_clearance_reward_sum(
            proximity_penalty_steepness = 5, # larger = steeper, more like a step function
            proximity_penalty_strength = 3, # larger = more influence on the reward function (should always be positive)
            safety_radius_coefficient = 2
        )
        # # Combine penalties and rewards
        reward = (
            1 + 10 * reaching_reward  # Reward for getting closer to the goal
            + clearance_reward_sum       # Reward for maintaining clearance
            + obstacle_penalty_sum       # Proximity and collision penalties
        )

        # Additional penalties for stability and control
        lin_vel_z_l2 = torch.square(self.agent.robot.root_linear_velocity[:, 2])
        ang_vel_xy_l2 = (
            torch.square(self.agent.robot.root_angular_velocity[:, :2])
        ).sum(axis=1)
        control_penalties = (
            lin_vel_z_l2 * -2
            + ang_vel_xy_l2 * -0.05
            + self._compute_undesired_contacts() * -1
            + torch.linalg.norm(self.agent.robot.qpos - self.default_qpos, axis=1)
            * -0.05
        )

        reward += control_penalties  # Add control penalties
        reward[info["fail"]] = 0  # Zero reward if the robot fails

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def generate_random_cube_positions(robot_x_distance_to_goal, num_cubes=5):
        positions = []
        
        max_forward_deviation = robot_x_distance_to_goal - QuadrupedReachEnv.CUBE_HALF_SIZE * 6
        def is_far_enough(new_pos):
            """Check if the new position is far enough from all existing positions."""
            for pos in positions:
                if math.dist(new_pos, pos) < QuadrupedReachEnv.MIN_DISTANCE_BETWEEN_CUBES:
                    # print(f'new position {new_pos} was not far enough from {positions}')
                    return False
            return True

        for _ in range(num_cubes):
            attempts_count = 0
            while attempts_count < 20:
                cube_forward_delta = max(random.random() * max_forward_deviation + QuadrupedReachEnv.CUBE_HALF_SIZE * 3, QuadrupedReachEnv.CUBE_HALF_SIZE)
                cube_horizontal_delta = random.random() * (QuadrupedReachEnv.MAX_CUBE_HORIZONTAL_DEVIATION * 2) - QuadrupedReachEnv.MAX_CUBE_HORIZONTAL_DEVIATION

                new_position = (cube_forward_delta, cube_horizontal_delta)

                if is_far_enough(new_position) or attempts_count >= attempts_count:
                    positions.append(new_position)
                    break
                attempts_count += 1

        return positions

@register_env("Cinnamon-Reach-v2", max_episode_steps=200)
class AnymalCReachEnv(QuadrupedReachEnv):
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )


@register_env("UnitreeGo2-Reach-v1", max_episode_steps=200)
class UnitreeGo2ReachEnv(QuadrupedReachEnv):
    _UNDESIRED_CONTACT_LINK_NAMES = ["FR_thigh", "RR_thigh", "FL_thigh", "RL_thigh"]

    def __init__(self, *args, robot_uids="unitree_go2_simplified_locomotion", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(
            UnitreeGo2Simplified.keyframes["standing"].qpos
        ).to(self.device)