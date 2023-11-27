import dataclasses
from typing import Any, Callable, Sequence

from flax import linen
import jax
import jax.numpy as jnp



ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(
                    hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden
    

class AtariTorso(linen.Module):
    """ConvNet Feature Extractor."""
    layer_sizes: Sequence[int] = (512,)
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.orthogonal(jnp.sqrt(2))
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = jnp.moveaxis(data, -3, -1) # jnp.transpose(data, (0, 2, 3, 1))
        hidden = hidden / (255.0)
        hidden = linen.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        hidden = self.activation(hidden)
        hidden = linen.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        hidden = self.activation(hidden)
        hidden = linen.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        hidden = self.activation(hidden)
        hidden = hidden.reshape(hidden.shape[:-3] + (-1,))
        hidden = linen.Dense(512, 
                             kernel_init=self.kernel_init, 
                             bias_init=jax.nn.initializers.constant(0.0)
                             )(hidden)
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                bias_init=jax.nn.initializers.constant(0.0),
                use_bias=self.bias)(
                    hidden)
            hidden = self.activation(hidden)
        return hidden
    

def make_atari_feature_extractor(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu
) -> FeedForwardNetwork:
    """Creates a CNN feature extractor."""
    feature_extractor = AtariTorso(
        layer_sizes=list(hidden_layer_sizes),
        activation=activation,
    )

    def apply(policy_params, obs):
        return feature_extractor.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)
    return FeedForwardNetwork(
        init=lambda key: feature_extractor.init(key, dummy_obs), apply=apply)

def make_policy_network(
        param_size: int,
        obs_size: int,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform())

    def apply(policy_params, obs):
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
    """Creates a policy network."""
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform())

    def apply(policy_params, obs):
        return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply)