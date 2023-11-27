from typing import Any, Mapping, Protocol, Tuple, TypeVar

import jax.numpy as jnp



NetworkType = TypeVar('NetworkType')


class Policy(Protocol):

  def __call__(
      self,
      observation: jnp.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
    pass
  

class NetworkFactory(Protocol[NetworkType]):

  def __call__(
      self,
      observation_size: int,
      action_size: int,
      preprocess_observations_fn,
  ) -> NetworkType:
    pass