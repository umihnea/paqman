from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: np.uint8
    reward: np.float32
    next_state: np.ndarray
    done: np.bool = np.bool(0)


@dataclass
class RankedTransition(Transition):
    error: np.float32 = None

    def __le__(self, other):
        if isinstance(other, RankedTransition):
            return self.error < other.error
        else:
            raise NotImplementedError(
                "Trying to compare ordered transition to non-ordered."
            )
