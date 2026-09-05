"""The trajectory golden on the numpy core: torch's two training runs, step by step.

Both silent bugs of 4 September passed every single-op case and showed only in a loop;
this holds the core's loops to torch's. Tolerances: the loss at every step within one
part in a hundred of torch's (measured drift: 2e-4 for the head, 2.5e-3 for the CNN
at step 40, float32 accumulation), and the final predictions identical.
"""
import pytest

import borch

from trajectory import RECIPES, check

TOL = 1e-2


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_the_core_walks_torchs_training_curve_and_lands_on_its_predictions(name):
    got = check(borch, name)
    assert got["worst_rel"] < TOL, (
        f"{name}: step {got['worst_step']} — torch {got['loss_torch']:.6f}, core {got['loss_here']:.6f} "
        f"({got['worst_rel']:.2e} relative)")
    assert got["pred_agree"] == 1.0, f"{name}: predictions agree on {got['pred_agree']:.0%} of the rows"
