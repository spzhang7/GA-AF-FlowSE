import pytest

from rl.grpo.official_evaluation import milestone_checkpoint_for_step


def _config():
    return {
        "run": {"collections": 1250},
        "collection": {"optimizer_updates": 4},
        "evaluation": {"selection_milestones": [0, 20, 40, 60, 80, 100]},
    }


def test_step_4000_maps_to_the_80pct_milestone(tmp_path):
    checkpoint, percentage, collection = milestone_checkpoint_for_step(
        _config(), run_dir=tmp_path, optimizer_step=4000
    )
    assert percentage == 80
    assert collection == 1000
    assert checkpoint.name == "checkpoint_milestone_080pct.pt"


@pytest.mark.parametrize("step", [3999, 3556, 5001])
def test_non_milestone_step_is_rejected(tmp_path, step):
    with pytest.raises(ValueError):
        milestone_checkpoint_for_step(
            _config(), run_dir=tmp_path, optimizer_step=step
        )
