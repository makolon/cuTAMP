# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from collections.abc import Mapping
from typing import Optional

import torch
from curobo.scene import Cuboid
from curobo.types import Pose

from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.samplers import grasp_4dof_sampler, grasp_6dof_sampler, place_4dof_sampler, sample_yaw

from cutamp.stream_initializers import (
    get_stream_data,
    grasp_data_to_actions,
    place_data_to_actions,
    push_data_to_actions,
    sample_initializer_indices,
)
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)

_log = logging.getLogger(__name__)


class ParticleInitializer:
    def __init__(
        self,
        world: TAMPWorld,
        config: TAMPConfiguration,
        stream_initializers: Optional[Mapping[str, object]] = None,
    ):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4 and config.place_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF placement supported for now, not {config.place_dof}")
        if config.grasp_dof != 4 and config.grasp_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported for now, not {config.grasp_dof}")
        if config.push_dof != 4 and config.push_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF push supported for now, not {config.push_dof}")
        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)

        # Load stream initializer data
        self.grasp_streams = get_stream_data(stream_initializers, "grasp")
        self.place_streams = get_stream_data(stream_initializers, "place")
        self.push_streams = get_stream_data(stream_initializers, "push")

        # Sampler caching
        self.pick_cache = {}
        self.place_cache = {}
        self.push_cache = {}

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = self.config.num_particles
        world = self.world
        particles = {"q0": self.q_init.clone()}
        deferred_params = set()
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        # Iterate through each ground operator in the plan skeleton and initialize and build up particles
        for idx, ground_op in enumerate(plan_skeleton):
            op_name = ground_op.operator.name
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"

            # MoveFree
            if op_name == MoveFree.name:
                q_start, _traj, q_end = params
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # MoveHolding
            elif op_name == MoveHolding.name:
                obj, grasp, q_start, _traj, q_end = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Pick
            elif op_name == Pick.name:
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Note: pick cache currently assumes object is at same pose as when sampled
                if obj in self.pick_cache:
                    # important, we need to clone here
                    particles[grasp] = self.pick_cache[obj]["sampled_grasps"].clone()
                    particles[q] = self.pick_cache[obj]["q_solutions"].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached grasp poses for {obj}. {num_particles}/{num_particles} success"
                    )
                    particles[f"{grasp}_confidences"] = self.pick_cache[obj]["confidences"]
                    continue

                stream_data = self.grasp_streams.get(obj)
                if stream_data is not None:
                    grasps_obj = stream_data.get("grasps_obj")
                    if grasps_obj is None:
                        raise ValueError(f"Grasp stream initializer for object '{obj}' is missing 'grasps_obj'")
                    confidences_pt = stream_data.get("confidences_pt")

                    grasp_actions, grasp_transforms = grasp_data_to_actions(grasps_obj, config.grasp_dof)
                    indices = sample_initializer_indices(
                        grasp_actions.shape[0],
                        num_particles,
                        device=grasp_actions.device,
                        scores=confidences_pt if isinstance(confidences_pt, torch.Tensor) else None,
                    )
                    particles[grasp] = grasp_actions[indices].clone()
                    if isinstance(confidences_pt, torch.Tensor):
                        particles[f"{grasp}_confidences"] = confidences_pt[indices].clone()
                    grasp_transforms = grasp_transforms[indices]
                    source = "External"
                else:
                    obj_curobo = world.get_object(obj)
                    obj_spheres = world.get_collision_spheres(obj)
                    num_samples = num_particles * 2
                    num_faces = 4 if isinstance(obj_curobo, Cuboid) else None
                    if config.grasp_dof == 4:
                        sampled_grasps = grasp_4dof_sampler(num_samples, obj_curobo, obj_spheres, num_faces=num_faces)
                        sampled_transforms = action_4dof_to_mat4x4(sampled_grasps)
                    else:
                        sampled_grasps = grasp_6dof_sampler(num_samples, obj_curobo, obj_spheres, num_faces=num_faces)
                        sampled_transforms = action_6dof_to_mat4x4(sampled_grasps)

                    # Filter grasps by collision with object
                    grasp_spheres = transform_spheres(world.robot_container.gripper_spheres, sampled_transforms)
                    grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres, activation_distance=0.0)

                    # Select grasps: use collision-free only, or fall back to lowest collision
                    collision_free_mask = grasp_coll <= 1e-2
                    if collision_free_mask.any():
                        selected_grasps = sampled_grasps[collision_free_mask][:num_particles]
                    else:
                        best_idxs = grasp_coll.topk(num_particles, largest=False).indices
                        selected_grasps = sampled_grasps[best_idxs]

                    if selected_grasps.shape[0] < num_particles:
                        sample_idxs = torch.randint(
                            0,
                            selected_grasps.shape[0],
                            (num_particles,),
                            device=selected_grasps.device,
                        )
                        selected_grasps = selected_grasps[sample_idxs]

                    particles[grasp] = selected_grasps
                    grasp_transforms = (
                        action_4dof_to_mat4x4(particles[grasp])
                        if config.grasp_dof == 4
                        else action_6dof_to_mat4x4(particles[grasp])
                    )
                    source = "Heuristic"

                world_from_obj = Pose.from_list(world.get_object(obj).pose).get_matrix()[0].to(world.device)
                world_from_grasp = world_from_obj @ grasp_transforms
                world_from_ee = world_from_grasp @ world.tool_from_ee

                # Solve IK with cuRobo
                ik_result = world.solve_pose(
                    world_from_ee,
                    return_seeds=1,
                )
                success, q_next = ik_result.success, ik_result.solution
                num_success = int(success.sum().item())
                log_debug(
                    f"{header}. {source} grasp IK success: "
                    f"{num_success}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                if num_success == 0:
                    log_debug(f"{header}. Pick IK failed for all {num_particles} particles; failing subgraph")
                    return None
                if num_success < num_particles:
                    log_debug(
                        f"{header}. Carrying forward previous configuration for "
                        f"{num_particles - num_success}/{num_particles} failed grasp IK particles"
                    )
                particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                if config.cache_subgraphs:
                    self.pick_cache[obj] = {
                        "sampled_grasps": particles[grasp].clone(),
                        "q_solutions": particles[q].clone(),
                        "confidences": particles.get(f"{grasp}_confidences"),
                    }

            # Place
            elif op_name == Place.name:
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if torch.equal(actual_grasp, cached_grasp):
                        # important, we need to clone here
                        sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                        particles[placement] = sampled_placements
                        particles[q] = self.place_cache[(obj, surface)]["q_solutions"].clone()
                        deferred_params.remove(q)
                        log_debug(
                            f"{header}. Using cached placement poses for {obj}. {num_particles}/{num_particles} success"
                        )
                        continue
                    log_debug(f"{header}. Cached grasp mismatch for {obj} on {surface}; resampling placements")

                place_stream_data = self.place_streams.get(obj)
                stream_data = place_stream_data.get(surface) if place_stream_data is not None else None
                if stream_data is not None:
                    placements_world = stream_data.get("placements_world")
                    if placements_world is None:
                        raise ValueError(
                            f"Placement stream initializer for object '{obj}' on surface '{surface}' "
                            "is missing 'placements_world'"
                        )
                    support_scores_pt = stream_data.get("support_scores_pt")

                    sampled_placements, world_from_obj = place_data_to_actions(placements_world)
                    indices = sample_initializer_indices(
                        sampled_placements.shape[0],
                        num_particles,
                        device=sampled_placements.device,
                        scores=support_scores_pt if isinstance(support_scores_pt, torch.Tensor) else None,
                    )
                    sampled_placements = sampled_placements[indices].clone()
                    world_from_obj = world_from_obj[indices]
                    source = "External"
                else:
                    obj_curobo = world.get_object(obj)
                    obj_spheres = world.get_collision_spheres(obj)
                    if config.random_init:
                        yaw = sample_yaw(num_particles * 2, None, world.device)
                        aabb = world.world_aabb.clone()
                        aabb[0, 2] = 0.0
                        aabb[1, 2] = max(aabb[1, 2], 0.2)
                        xyz = sample_between_bounds(num_particles * 2, aabb)
                        sampled_placements = torch.cat([xyz, yaw.to(xyz.dtype).unsqueeze(-1)], dim=1)
                    else:
                        sampled_placements = place_4dof_sampler(
                            num_particles * 2,
                            obj_curobo,
                            obj_spheres,
                            world.get_object(surface),
                            surface_rep=config.placement_check,
                            shrink_dist=config.placement_shrink_dist,
                            collision_activation_dist=config.world_activation_distance,
                        )

                    world_from_obj = action_4dof_to_mat4x4(sampled_placements)
                    obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                    place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                    best_idxs = place_coll.topk(num_particles, largest=False).indices
                    sampled_placements = sampled_placements[best_idxs].clone()
                    world_from_obj = world_from_obj[best_idxs]
                    source = "Heuristic"
                particles[placement] = sampled_placements

                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])

                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    ik_result = world.solve_pose(
                        world_from_ee,
                        return_seeds=1,
                    )
                    success, q_next = ik_result.success, ik_result.solution
                    num_success = int(success.sum().item())
                    log_debug(
                        f"{header}. {source} place IK success: "
                        f"{num_success}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                if config.cache_subgraphs and not config.random_init:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": particles[placement].clone(),
                        "q_solutions": particles[q].clone(),
                        "grasp": particles[grasp].clone(),
                    }

            # Push
            elif op_name == Push.name:
                button, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if button in self.push_cache:
                    # important, we need to clone here
                    sampled_push = self.push_cache[button]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    particles[q] = self.push_cache[button]["q_solutions"].clone()
                    deferred_params.remove(q)
                    log_debug(f"{header}. Using cached push poses for {button}. {num_particles}/{num_particles} success")
                    continue

                stream_data = self.push_streams.get(button)
                if stream_data is not None:
                    pushes_world = stream_data.get("pushes_world")
                    if pushes_world is None:
                        raise ValueError(f"Push stream initializer for button '{button}' is missing 'pushes_world'")
                    push_scores_pt = stream_data.get("push_scores_pt")

                    sampled_push, _world_from_button = push_data_to_actions(pushes_world, config.push_dof)
                    indices = sample_initializer_indices(
                        sampled_push.shape[0],
                        num_particles,
                        device=sampled_push.device,
                        scores=push_scores_pt if isinstance(push_scores_pt, torch.Tensor) else None,
                    )
                    sampled_push = sampled_push[indices].clone()
                else:
                    aabb = world.get_aabb(button).clone()
                    lower_xy, upper_xy = aabb[:, :2]
                    lower_xy = lower_xy + 0.01
                    upper_xy = upper_xy - 0.01
                    sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.device) * (upper_xy - lower_xy)
                    sampled_z = aabb[1, 2].expand(num_particles) + 0.02 + config.world_activation_distance
                    sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.device)
                    sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)
                particles[push_pose] = sampled_push

                world_from_push = (
                    action_4dof_to_mat4x4(sampled_push)
                    if config.push_dof == 4
                    else action_6dof_to_mat4x4(sampled_push)
                )
                world_from_ee = world_from_push @ world.tool_from_ee

                # Solve IK with cuRobo
                ik_result = world.solve_pose(
                    world_from_ee,
                    return_seeds=1,
                )
                success, q_next = ik_result.success, ik_result.solution
                num_success = int(success.sum().item())
                log_debug(
                    f"{header}. IK success: {num_success}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Cache the push poses
                if config.cache_subgraphs:
                    self.push_cache[button] = {
                        "sampled_push": particles[push_pose].clone(),
                        "q_solutions": particles[q].clone(),
                    }

            # Unknown
            else:
                raise NotImplementedError(f"Unsupported operator {op_name}")

        # There should not be any deferred parameters left
        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
