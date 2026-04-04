import tempfile
import unittest
from collections import defaultdict, deque
from pathlib import Path

import pybullet as pb
import torch

from cutamp.algorithm import CutampRunResult
from cutamp.clients.vlm_client import BaseVLMClient
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.book_shelf import load_book_shelf_env
from cutamp.envs.mini_kitchen import load_mini_kitchen_env
from cutamp.envs.utils import clone_tamp_environment, reduce_tamp_environment, set_object_pose, set_openable_state
from cutamp.sim.pybullet_scene import build_pybullet_scene, disconnect_pybullet_scene
from cutamp.sim.pybullet_sync import sync_pybullet_scene_from_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import HandEmpty, In, On, Open, get_initial_state, get_tamp_operators_for_env
from cutamp.task_planning import task_plan_generator
from cutamp.vlm_tamp import (
    build_scene_abstraction,
    check_subgoal_achieved,
    get_env_adapter,
    parse_formal_subgoals,
    render_vlm_scene,
    render_simple_annotated_scene,
    run_vlm_tamp,
)


def _make_config(experiment_root: str) -> TAMPConfiguration:
    return TAMPConfiguration(
        enable_vlm_tamp=True,
        open_goal="put both books on the shelf",
        experiment_root=experiment_root,
        enable_visualizer=False,
        save_retrieval_artifacts=False,
    )


def _support_pose(surface, obj, buffer: float = 0.01):
    surface_top = surface.pose[2] + surface.dims[2] / 2
    obj_height = obj.dims[2]
    return [surface.pose[0], surface.pose[1], surface_top + obj_height / 2 + buffer, *obj.pose[3:]]


class _ScriptedVLMClient(BaseVLMClient):
    def __init__(self, responses: dict[str, list[str]]):
        self._stage_queues = defaultdict(deque)
        for stage, stage_responses in responses.items():
            self._stage_queues[stage].extend(stage_responses)

    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        if self._stage_queues[stage]:
            return self._stage_queues[stage].popleft()
        raise RuntimeError(f"No scripted VLM response available for stage '{stage}'")


class _AlwaysSuccessPlanner:
    def __call__(self, env, config, cost_reducer, constraint_checker, q_init=None, experiment_id=None):
        final_env = clone_tamp_environment(env)
        goal_atom = next(atom for atom in env.goal_state if atom.name != HandEmpty.name)
        if goal_atom.name == On.name:
            obj_name, surface_name = goal_atom.values
            obj = next(obj for obj in final_env.movables if obj.name == obj_name)
            surface = next(obj for obj in final_env.statics if obj.name == surface_name)
            set_object_pose(final_env, obj_name, _support_pose(surface, obj))
        elif goal_atom.name == Open.name:
            set_openable_state(final_env, goal_atom.values[0], True)
        elif goal_atom.name == In.name:
            obj_name, container_name = goal_atom.values
            obj = next(obj for obj in final_env.movables if obj.name == obj_name)
            interior_name = final_env.metadata["openables"][container_name]["interior"]
            interior = next(obj_ for obj_ in final_env.statics if obj_.name == interior_name)
            set_object_pose(final_env, obj_name, _support_pose(interior, obj, buffer=0.0))
        return CutampRunResult(
            curobo_plan=None,
            num_satisfying_final=1,
            found_solution=True,
            best_particle={"q0": torch.zeros(1)},
            final_rollout=None,
            final_env=final_env,
            final_q_init=torch.zeros(7),
            final_plan_skeleton=["mock"],
            overall_metrics={"found_solution": True},
            timer_summaries={},
            collision_summary={},
        )


class _BlockerPlanner:
    def __call__(self, env, config, cost_reducer, constraint_checker, q_init=None, experiment_id=None):
        movable_names = {obj.name for obj in env.movables}
        goal_atom = next(atom for atom in env.goal_state if atom.name == On.name)
        obj_name, _ = goal_atom.values
        if obj_name == "book_green" and movable_names == {"book_green"}:
            return CutampRunResult(
                curobo_plan=None,
                num_satisfying_final=0,
                found_solution=False,
                best_particle=None,
                final_rollout=None,
                final_env=None,
                final_q_init=None,
                final_plan_skeleton=None,
                overall_metrics={"found_solution": False},
                timer_summaries={},
                collision_summary={"book_green_to_book_blue": 1.0, "robot_to_world": 0.1},
            )
        return _AlwaysSuccessPlanner()(env, config, cost_reducer, constraint_checker, q_init=q_init, experiment_id=experiment_id)


class VLMTAMPTests(unittest.TestCase):
    def setUp(self):
        self.env = load_book_shelf_env()
        self.book_shelf_adapter = get_env_adapter(self.env)
        self.mini_kitchen_env = load_mini_kitchen_env()
        self.mini_kitchen_adapter = get_env_adapter(self.mini_kitchen_env)
        self.cost_reducer = CostReducer(default_constraint_to_mult.copy())
        self.constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    def test_render_simple_annotated_scene_generates_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = build_scene_abstraction(self.env)
            image_path = render_simple_annotated_scene(
                self.env,
                Path(tmpdir) / "scene.png",
                open_goal="put both books on the shelf",
                scene_text=scene["scene_text"],
            )
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)

    def test_parse_formal_subgoals_filters_invalid_and_duplicates(self):
        achieved = {On.ground("book_blue", "shelf")}
        parsed, errors = parse_formal_subgoals(
            self.book_shelf_adapter,
            "On(book_blue, shelf)\nOn(book_green, shelf)\nOn(book_green, shelf)\nMove(book_blue, shelf)",
            self.env,
            achieved_atoms=achieved,
            hand_empty=True,
        )
        self.assertEqual([item.canonical for item in parsed], ["On(book_green, shelf)"])
        self.assertEqual(errors, [])

    def test_parse_formal_subgoals_reports_unknown_objects(self):
        parsed, errors = parse_formal_subgoals(
            self.book_shelf_adapter,
            "On(book_purple, shelf)",
            self.env,
            achieved_atoms=set(),
            hand_empty=True,
        )
        self.assertEqual(parsed, [])
        self.assertTrue(any("Unknown movable object" in error for error in errors))

    def test_parse_formal_subgoals_extracts_kitchen_atoms_from_verbose_text(self):
        parsed, errors = parse_formal_subgoals(
            self.mini_kitchen_adapter,
            "First Open(drawer). Then In(mug, cabinet). Finally place it as On(mug, counter).",
            self.mini_kitchen_env,
            achieved_atoms=set(),
            hand_empty=True,
        )
        self.assertEqual(
            [item.canonical for item in parsed],
            ["Open(drawer)", "In(mug, cabinet)", "On(mug, counter)"],
        )
        self.assertEqual(errors, [])

    def test_mini_kitchen_loader_has_openables_and_initial_atoms(self):
        env = self.mini_kitchen_env
        self.assertIn("openables", env.metadata)
        self.assertEqual({obj.name for obj in env.type_to_objects["Container"]}, {"cabinet", "drawer"})
        self.assertEqual({obj.name for obj in env.type_to_objects["Openable"]}, {"cabinet", "drawer"})
        self.assertIn(In.ground("mug", "cabinet"), env.initial_atoms)
        self.assertIn(In.ground("spoon", "drawer"), env.initial_atoms)

    def test_check_subgoal_achieved_handles_open_and_in(self):
        env = clone_tamp_environment(self.mini_kitchen_env)
        self.assertTrue(check_subgoal_achieved(env, In.ground("mug", "cabinet"), hand_empty=True))
        self.assertFalse(check_subgoal_achieved(env, Open.ground("cabinet"), hand_empty=True))
        set_openable_state(env, "cabinet", True)
        self.assertTrue(check_subgoal_achieved(env, Open.ground("cabinet"), hand_empty=True))

    def test_object_reduction_preserves_containers_and_openables(self):
        reduced = reduce_tamp_environment(self.mini_kitchen_env, movable_names=["mug"])
        self.assertEqual({obj.name for obj in reduced.movables}, {"mug"})
        self.assertEqual({obj.name for obj in reduced.type_to_objects["Container"]}, {"cabinet", "drawer"})
        self.assertEqual({obj.name for obj in reduced.type_to_objects["Openable"]}, {"cabinet", "drawer"})
        self.assertEqual({obj.name for obj in reduced.type_to_objects["Surface"]}, {"table", "counter", "cabinet_interior", "drawer_interior"})

    def test_object_reduction_can_leave_zero_movables_and_clone(self):
        reduced = reduce_tamp_environment(self.mini_kitchen_env, movable_names=[])
        cloned = clone_tamp_environment(reduced)
        self.assertEqual(cloned.movables, [])
        self.assertIn("Movable", cloned.type_to_objects)
        self.assertEqual(cloned.type_to_objects["Movable"], [])

    def test_pybullet_scene_sync_updates_openable_pose(self):
        env = clone_tamp_environment(self.mini_kitchen_env)
        scene = build_pybullet_scene(env)
        sync_pybullet_scene_from_env(scene, env)
        cabinet_joint = scene.openable_joints["cabinet"]["joints"][0]
        closed_value = pb.getJointState(
            scene.body_ids[cabinet_joint["body_name"]],
            cabinet_joint["joint_index"],
            physicsClientId=scene.client_id,
        )[0]
        set_openable_state(env, "cabinet", True)
        sync_pybullet_scene_from_env(scene, env)
        open_value = pb.getJointState(
            scene.body_ids[cabinet_joint["body_name"]],
            cabinet_joint["joint_index"],
            physicsClientId=scene.client_id,
        )[0]
        disconnect_pybullet_scene(scene)
        self.assertNotEqual(closed_value, open_value)
        self.assertLess(open_value, closed_value)

    def test_pybullet_scene_sync_updates_movable_pose(self):
        env = clone_tamp_environment(self.mini_kitchen_env)
        scene = build_pybullet_scene(env)
        sync_pybullet_scene_from_env(scene, env)
        set_object_pose(env, "mug", [0.55, -0.05, 0.08, *env.movables[0].pose[3:]])
        sync_pybullet_scene_from_env(scene, env)
        mug_pos, _ = pb.getBasePositionAndOrientation(scene.body_ids["mug"], physicsClientId=scene.client_id)
        disconnect_pybullet_scene(scene)
        self.assertAlmostEqual(mug_pos[0], 0.55, places=3)
        self.assertAlmostEqual(mug_pos[1], -0.05, places=3)

    def test_render_vlm_scene_pybullet_rgb_writes_mosaic_and_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = build_scene_abstraction(self.mini_kitchen_env)
            image_path = render_vlm_scene(
                self.mini_kitchen_env,
                Path(tmpdir) / "query_00.png",
                "pybullet_rgb",
                open_goal="open the cabinet and place the mug on the counter",
                scene_text=scene["scene_text"],
            )
            self.assertTrue(image_path.exists())
            self.assertTrue((Path(tmpdir) / "query_00_view0.png").exists())
            self.assertTrue((Path(tmpdir) / "query_00_view1.png").exists())
            self.assertTrue((Path(tmpdir) / "query_00_camera_meta.json").exists())
            self.assertTrue((Path(tmpdir) / "query_00.png").exists())

    def test_task_planner_uses_open_before_pick_from_container(self):
        env = self.mini_kitchen_env
        initial_state = get_initial_state(
            movables=[obj.name for obj in env.movables],
            surfaces=[obj.name for obj in env.type_to_objects["Surface"]],
            containers=[obj.name for obj in env.type_to_objects["Container"]],
            openables=[obj.name for obj in env.type_to_objects["Openable"]],
        )
        initial_state = frozenset(set(initial_state).union(env.initial_atoms))
        goal_state = frozenset({On.ground("mug", "counter"), HandEmpty.ground()})
        plan = next(
            task_plan_generator(
                initial_state,
                goal_state,
                operators=get_tamp_operators_for_env("mini_kitchen"),
                max_depth=12,
            )
        )
        op_names = [op.name for op in plan]
        self.assertIn("Open", op_names)
        self.assertIn("PickFromContainer", op_names)
        self.assertLess(op_names.index("Open"), op_names.index("PickFromContainer"))

    def test_task_planner_can_solve_open_drawer_subgoal(self):
        env = self.mini_kitchen_env
        initial_state = get_initial_state(
            movables=[obj.name for obj in env.movables],
            surfaces=[obj.name for obj in env.type_to_objects["Surface"]],
            containers=[obj.name for obj in env.type_to_objects["Container"]],
            openables=[obj.name for obj in env.type_to_objects["Openable"]],
        )
        initial_state = frozenset(set(initial_state).union(env.initial_atoms))
        goal_state = frozenset({Open.ground("drawer"), HandEmpty.ground()})
        plan = next(
            task_plan_generator(
                initial_state,
                goal_state,
                operators=get_tamp_operators_for_env("mini_kitchen"),
                max_depth=12,
            )
        )
        self.assertEqual([op.name for op in plan], ["MoveFree", "Open"])

    def test_run_vlm_tamp_reprompts_after_invalid_subgoal_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            client = _ScriptedVLMClient(
                {
                    "stage1": [
                        "Move the blue book to the shelf",
                        "Place the blue book on the shelf",
                    ],
                    "stage2": [
                        "Move(book_blue, shelf)",
                        "On(book_blue, shelf)\nHandEmpty()",
                    ],
                }
            )
            result = run_vlm_tamp(
                self.env,
                config,
                self.cost_reducer,
                self.constraint_checker,
                experiment_id="reprompt_case",
                vlm_client=client,
                planner_fn=_AlwaysSuccessPlanner(),
            )
            self.assertTrue(result.found_solution)
            self.assertEqual(result.subgoals_validated, ["On(book_blue, shelf)"])
            self.assertEqual(len(result.query_trace), 2)

    def test_run_vlm_tamp_promotes_blocker_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            client = _ScriptedVLMClient(
                {
                    "stage1": ["Put the green book on the shelf"],
                    "stage2": ["On(book_green, shelf)"],
                }
            )
            result = run_vlm_tamp(
                self.env,
                config,
                self.cost_reducer,
                self.constraint_checker,
                experiment_id="blocker_case",
                vlm_client=client,
                planner_fn=_BlockerPlanner(),
            )
            self.assertTrue(result.found_solution)
            modes = [attempt["mode"] for attempt in result.attempt_trace if "mode" in attempt]
            self.assertEqual(modes[:2], ["reduced", "reduced_with_blockers"])

    def test_run_vlm_tamp_keeps_full_environment_across_reduced_subgoals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            client = _ScriptedVLMClient(
                {
                    "stage1": ["Put the blue and green books on the shelf"],
                    "stage2": ["On(book_blue, shelf)\nOn(book_green, shelf)"],
                }
            )
            result = run_vlm_tamp(
                self.env,
                config,
                self.cost_reducer,
                self.constraint_checker,
                experiment_id="multi_subgoal_case",
                vlm_client=client,
                planner_fn=_AlwaysSuccessPlanner(),
            )
            self.assertTrue(result.found_solution)
            self.assertEqual(result.subgoals_validated, ["On(book_blue, shelf)", "On(book_green, shelf)"])
            self.assertIsNotNone(result.final_env)
            self.assertEqual({obj.name for obj in result.final_env.movables}, {"book_blue", "book_green"})

    def test_run_vlm_tamp_on_mini_kitchen_open_and_place_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            config = TAMPConfiguration(
                **{
                    **config.__dict__,
                    "open_goal": "open the cabinet and place the mug on the counter",
                    "vlm_render_style": "pybullet_rgb",
                }
            )
            client = _ScriptedVLMClient(
                {
                    "stage1": ["Open the cabinet\nPut the mug on the counter"],
                    "stage2": ["Open(cabinet)\nOn(mug, counter)"],
                }
            )
            result = run_vlm_tamp(
                self.mini_kitchen_env,
                config,
                self.cost_reducer,
                self.constraint_checker,
                experiment_id="mini_kitchen_case",
                vlm_client=client,
                planner_fn=_AlwaysSuccessPlanner(),
            )
            self.assertTrue(result.found_solution)
            self.assertEqual(result.subgoals_validated, ["Open(cabinet)", "On(mug, counter)"])
            exp_dir = Path(result.experiment_dir) / "vlm"
            self.assertTrue((exp_dir / "query_00.png").exists())
            self.assertTrue((exp_dir / "query_00_view0.png").exists())
            self.assertTrue((exp_dir / "query_00_view1.png").exists())


if __name__ == "__main__":
    unittest.main()
