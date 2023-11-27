from typing import NamedTuple

import jax.numpy as jnp

NestedArray = jnp.ndarray

class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray