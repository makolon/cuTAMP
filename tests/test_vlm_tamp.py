import tempfile
import unittest
from pathlib import Path

import torch

from cutamp.algorithm import CutampRunResult
from cutamp.clients.vlm_client import StubVLMClient
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.book_shelf import load_book_shelf_env
from cutamp.envs.utils import clone_tamp_environment, set_object_pose
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import HandEmpty, On
from cutamp.vlm_tamp import (
    build_scene_abstraction,
    parse_formal_subgoals,
    render_simple_annotated_scene,
    run_vlm_tamp,
)


def _make_config(experiment_root: str) -> TAMPConfiguration:
    return TAMPConfiguration(
        enable_vlm_tamp=True,
        open_goal="put both books on the shelf",
        vlm_backend="stub",
        experiment_root=experiment_root,
        enable_visualizer=False,
        save_retrieval_artifacts=False,
    )


def _support_pose(surface, obj, buffer: float = 0.01):
    surface_top = surface.pose[2] + surface.dims[2] / 2
    obj_height = obj.dims[2]
    return [surface.pose[0], surface.pose[1], surface_top + obj_height / 2 + buffer, *obj.pose[3:]]


class _AlwaysSuccessPlanner:
    def __call__(self, env, config, cost_reducer, constraint_checker, q_init=None, experiment_id=None):
        final_env = clone_tamp_environment(env)
        goal_atom = next(atom for atom in env.goal_state if atom.name == On.name)
        obj_name, surface_name = goal_atom.values
        obj = next(obj for obj in final_env.movables if obj.name == obj_name)
        surface = next(obj for obj in final_env.statics if obj.name == surface_name)
        set_object_pose(final_env, obj_name, _support_pose(surface, obj))
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
            "On(book_blue, shelf)\nOn(book_green, shelf)\nOn(book_green, shelf)\nMove(book_blue, shelf)",
            self.env,
            achieved_atoms=achieved,
            hand_empty=True,
        )
        self.assertEqual([item.canonical for item in parsed], ["On(book_green, shelf)"])
        self.assertTrue(any("Unsupported predicate" in error for error in errors))

    def test_run_vlm_tamp_reprompts_after_invalid_subgoal_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            client = StubVLMClient(
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
            client = StubVLMClient(
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


if __name__ == "__main__":
    unittest.main()
