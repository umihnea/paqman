import pytest

from deepq.checkpoint_manager import CheckpointManager


def test_checkpoint_manager():
    manager = CheckpointManager(
        {"path": "../data/checkpoints", "max_size": 3, "every": 1000}
    )

    manager.add({"a": 1}, 100, 1, commit=False)
    manager.add({"b": 2}, 200, 2, commit=False)
    manager.add({"c": 3}, 500, 3, commit=False)

    # This should discard {'a': 1}
    manager.add({"d": 4}, 1000, 4, commit=False)
    assert len(manager.checkpoints) == 3
    assert manager.checkpoints[0].score == 200
    assert manager.checkpoints[2].score == 1000

    # This should be a no-op
    manager.add({"e": 5}, 150, 5, commit=False)
    assert len(manager.checkpoints) == 3
    assert manager.checkpoints[0].score == 200
