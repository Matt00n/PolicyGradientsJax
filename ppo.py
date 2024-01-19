"""Proximal Policy Optimization (PPO)."""

from absl import logging, app
from functools import partial
import os
import pickle
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

# os.environ[
#     "XLA_PYTHON_CLIENT_MEM_FRACTION"
# ] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs import make_env, Transition, has_discrete_action_space, is_atari_env
from networks.policy import Policy
from networks.networks import FeedForwardNetwork, ActivationFn, make_policy_network, make_value_network, make_atari_feature_extractor
from networks.distributions import NormalTanhDistribution, ParametricDistribution, PolicyNormalDistribution, DiscreteDistribution

class Config:
    # experiment
    experiment_name = 'ppo_test_video'
    seed = 20
    platform = 'cpu' # CPU or GPU
    capture_video = True # Not implemented
    write_logs_to_file = False
    save_model = False

    # environment
    env_id = 'Humanoid-v4' 
    num_envs = 8
    parallel_envs = True 
    clip_actions = False
    normalize_observations = True 
    normalize_rewards = True
    clip_observations = 10.
    clip_rewards = 10.
    eval_env = True
    num_eval_episodes = 10
    eval_every = 5
    deterministic_eval = True

    # algorithm hyperparameters
    total_timesteps = int(1e6) * 8 
    learning_rate = 3e-4 
    unroll_length = 2048 
    anneal_lr = True
    gamma = 0.99 
    gae_lambda = 0.95
    batch_size = 1 # number of unrolls per minibatch 
    num_minibatches = 8
    update_epochs = 10 
    normalize_advantages = True
    clip_eps = 0.2
    entropy_cost = 0.00 
    vf_cost = 0.5
    max_grad_norm = 0.5
    target_kl = None
    reward_scaling = 1. 
    
    # policy params
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4 
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5 
    activation: ActivationFn = nn.swish 
    squash_distribution: bool = True

    # atari params
    atari_dense_layer_sizes: Sequence[int] = (512,)


Metrics = Mapping[str, jnp.ndarray]

_PMAP_AXIS_NAME = 'i'


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # in order to avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


@flax.struct.dataclass
class PPONetworkParams:
    """Contains training state for the learner."""
    policy: Any
    value: Any


@flax.struct.dataclass
class PPONetworks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class AtariPPONetworkParams:
    """Contains training state for the learner."""
    feature_extractor: Any
    policy: Any
    value: Any


@flax.struct.dataclass
class AtariPPONetworks:
    feature_extractor: FeedForwardNetwork
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Union[PPONetworkParams, AtariPPONetworkParams]
    env_steps: jnp.ndarray


def make_inference_fn(ppo_networks: Union[PPONetworks, AtariPPONetworks]):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: Any,
                    deterministic: bool = False) -> Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        @jax.jit
        def policy(observations: jnp.ndarray,
                key_sample: jnp.ndarray) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            logits = policy_network.apply(params, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions)
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions
            }

        return policy

    return make_policy


def make_feature_extraction_fn(ppo_networks: AtariPPONetworks):
    """Creates feature extractor for inference."""

    def make_feature_extractor(params: Any):
        shared_feature_extractor = ppo_networks.feature_extractor

        @jax.jit
        def feature_extractor(observations: jnp.ndarray) -> jnp.ndarray:
            return shared_feature_extractor.apply(params, observations)

        return feature_extractor

    return make_feature_extractor


def make_ppo_networks(
        observation_size: int,
        action_size: int,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: ActivationFn = nn.swish,
        sqash_distribution: bool = True,
        discrete_policy: bool = False,
        shared_feature_extractor: bool = False,
        feature_extractor_dense_hidden_layer_sizes: Optional[Sequence[int]] = (512,),
    ) -> PPONetworks:
    """Make PPO networks with preprocessor."""
    if discrete_policy:
        parametric_action_distribution = DiscreteDistribution(
            param_size=action_size)
    elif sqash_distribution:
        parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size)
    else:
        parametric_action_distribution = PolicyNormalDistribution(
            event_size=action_size)
    if shared_feature_extractor:
        feature_extractor = make_atari_feature_extractor(
            obs_size=observation_size,
            hidden_layer_sizes=feature_extractor_dense_hidden_layer_sizes,
            activation=nn.relu
        )
        policy_network = make_policy_network(
            parametric_action_distribution.param_size,
            feature_extractor_dense_hidden_layer_sizes[-1],
            hidden_layer_sizes=(),
            activation=activation)
        value_network = make_value_network(
            feature_extractor_dense_hidden_layer_sizes[-1],
            hidden_layer_sizes=(),
            activation=activation)
        return AtariPPONetworks(
            feature_extractor=feature_extractor,
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=parametric_action_distribution)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation)
    value_network = make_value_network(
        observation_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation)

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
        truncation: A float32 tensor of shape [T, B] with truncation signal.
        termination: A float32 tensor of shape [T, B] with termination signal.
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
        lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount *
                    (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)



def compute_ppo_loss(
    params: Union[PPONetworkParams, AtariPPONetworkParams],
    data: Transition,
    rng: jnp.ndarray,
    ppo_network: Union[PPONetworks, AtariPPONetworks],
    vf_cost: float = 0.5,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    shared_feature_extractor: bool = False,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes PPO loss including value loss and entropy bonus.

    Policy loss: $L_\pi = \frac{1}{\lvert \mathcal{D} \rvert} \sum_{\mathcal{D}} 
    \min \biggl( \frac{\pi_\theta (a \mid s)}{\pi_\text{old} 
    (a \mid s)} \hat{A}, \text{clip}\Bigl( \frac{\pi_\theta (a \mid s)}{\pi_\text{old} 
    (a \mid s)}, 1-\varepsilon, 1+\varepsilon \Bigr) \hat{A} \biggr)$

    Args:
        params: Network parameters,
        data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
            ['policy_extras']['log_prob']
        rng: Random key
        ppo_network: PPO networks.
        entropy_cost: entropy cost.
        discounting: discounting,
        reward_scaling: reward multiplier.
        gae_lambda: General advantage estimation lambda.
        clipping_epsilon: Policy loss clipping epsilon
        normalize_advantage: whether to normalize advantage estimate
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    hidden_boot = data.next_observation[-1]
    if shared_feature_extractor:
        feature_extractor_apply = ppo_network.feature_extractor.apply
        hidden = feature_extractor_apply(params.feature_extractor, data.observation)
        hidden_boot = feature_extractor_apply(params.feature_extractor, 
                                          data.next_observation[-1])
    
    policy_logits = policy_apply(params.policy,
                                hidden)

    baseline = value_apply(params.value, hidden)

    
    bootstrap_value = value_apply(params.value,
                                    hidden_boot)

    rewards = data.reward * reward_scaling
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation) 

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action'])
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting)
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = target_action_log_probs - behaviour_action_log_probs
    rho_s = jnp.exp(log_ratio)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon,
                                1 + clipping_epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
    approx_kl = ((rho_s - 1) - log_ratio).mean()

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_cost

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': v_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy, 
        'approx_kl': jax.lax.stop_gradient(approx_kl), 
    }

    return total_loss, metrics




def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_id}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        from absl import flags
        flags.FLAGS.alsologtostderr = True
        log_path = f'./training_logs/ppo/{run_name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logging.get_absl_handler().use_absl_log_file('logs', log_path)

    logging.get_absl_handler().setFormatter(None)

    # jax set up devices
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    assert Config.num_envs % device_count == 0


    assert Config.batch_size * Config.num_minibatches % Config.num_envs == 0
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        Config.batch_size * Config.unroll_length * Config.num_minibatches)
    num_training_steps = np.ceil(Config.total_timesteps / env_step_per_training_step).astype(int)

    # log hyperparameters
    logging.info("|param: value|")
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            logging.info(f"|{key}:  {value}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    # handle / split random keys
    key = jax.random.PRNGKey(Config.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_envs, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, key_feature_extractor = jax.random.split(global_key, 3)
    del global_key
 
    is_atari = is_atari_env(Config.env_id)
    envs = make_env(
        env_id=Config.env_id,
        num_envs=Config.num_envs,
        parallel=Config.parallel_envs,
        clip_actions=Config.clip_actions,
        norm_obs=Config.normalize_observations,
        norm_reward=Config.normalize_rewards,
        clip_obs=Config.clip_observations,
        clip_rewards=Config.clip_rewards,
        is_atari=is_atari,
    )

    discrete_action_space = has_discrete_action_space(envs)
    envs.seed(int(key_envs[0]))
    env_state = envs.reset() 


    if discrete_action_space:
        action_size = envs.action_space.n
    else:
        action_size = np.prod(envs.action_space.shape) # flatten action size for nested spaces
    if is_atari:
        observation_shape = env_state.obs.shape[-3:]
    else:
        observation_shape = env_state.obs.shape[-1]

    ppo_network = make_ppo_networks(
        observation_size=observation_shape, # NOTE only works with flattened observation space
        action_size=action_size, # flatten action size for nested spaces
        policy_hidden_layer_sizes=Config.policy_hidden_layer_sizes, 
        value_hidden_layer_sizes=Config.value_hidden_layer_sizes,
        activation=Config.activation,
        sqash_distribution=Config.squash_distribution,
        discrete_policy=discrete_action_space,
        shared_feature_extractor=is_atari,
        feature_extractor_dense_hidden_layer_sizes=Config.atari_dense_layer_sizes,
    )
    make_policy = make_inference_fn(ppo_network)
    if is_atari:
        make_feature_extractor = make_feature_extraction_fn(ppo_network)

    # create optimizer
    if Config.anneal_lr:    
        learning_rate = optax.linear_schedule(
            Config.learning_rate, 
            Config.learning_rate * 0.01, # 0
            transition_steps=Config.total_timesteps, 
        )
    else:
        learning_rate = Config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.max_grad_norm),
        optax.adam(learning_rate),
    )

    # create loss function via functools.partial
    loss_fn = partial(
        compute_ppo_loss,
        ppo_network=ppo_network,
        vf_cost=Config.vf_cost,
        entropy_cost=Config.entropy_cost,
        discounting=Config.gamma,
        reward_scaling=Config.reward_scaling,
        gae_lambda=Config.gae_lambda,
        clipping_epsilon=Config.clip_eps,
        normalize_advantage=Config.normalize_advantages,
        shared_feature_extractor=is_atari,
    )


    def loss_and_pgrad(loss_fn: Callable[..., float],
                        pmap_axis_name: Optional[str],
                        has_aux: bool = False):
        g = jax.value_and_grad(loss_fn, has_aux=has_aux)

        def h(*args, **kwargs):
            value, grad = g(*args, **kwargs)
            return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

        return g if pmap_axis_name is None else h
    

    def gradient_update_fn(loss_fn: Callable[..., float],
                            optimizer: optax.GradientTransformation,
                            pmap_axis_name: Optional[str],
                            has_aux: bool = False):
        """Wrapper of the loss function that apply gradient updates.

        Args:
            loss_fn: The loss function.
            optimizer: The optimizer to apply gradients.
            pmap_axis_name: If relevant, the name of the pmap axis to synchronize
            gradients.
            has_aux: Whether the loss_fn has auxiliary data.

        Returns:
            A function that takes the same argument as the loss function plus the
            optimizer state. The output of this function is the loss, the new parameter,
            and the new optimizer state.
        """
        loss_and_pgrad_fn = loss_and_pgrad(
            loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

        def f(*args, optimizer_state):
            value, grads = loss_and_pgrad_fn(*args)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], params_update)
            return value, params, optimizer_state

        return f
    
    gradient_update_fn = gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    # minibatch training step
    def minibatch_step(carry, data: Transition,):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics


    # sgd step
    def sgd_step(carry, unused_t, data: Transition):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step, 
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (optimizer_state, params, key), metrics
    

    # learning 
    def learn(
        data: Transition,
        training_state: TrainingState,
        key_sgd: jnp.ndarray,
    ):
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(
                sgd_step, data=data),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_steps=training_state.env_steps + env_step_per_training_step)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return new_training_state, metrics
    
    learn = jax.pmap(learn, axis_name=_PMAP_AXIS_NAME)


    # initialize params & training state
    if is_atari:
        init_params = AtariPPONetworkParams(
            feature_extractor=ppo_network.feature_extractor.init(key_feature_extractor),
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    else:
        init_params = PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])
    

    # create eval env
    if Config.eval_env:
        eval_env = make_env(
            env_id=Config.env_id,
            num_envs=1, 
            parallel=False, 
            norm_obs=False,
            norm_reward=False,
            clip_obs=Config.clip_observations,
            clip_rewards=Config.clip_rewards,
            evaluate=True,
            capture_video=Config.capture_video,
        )
        eval_env.seed(int(eval_key[0]))
        eval_state = eval_env.reset()


    # initialize metrics
    global_step = 0
    start_time = time.time()
    training_walltime = 0
    scores = []

    # training loop
    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()

        new_key, local_key = jax.random.split(local_key)
        training_state, env_state = _strip_weak_type((training_state, env_state))
        key_sgd, key_generate_unroll = jax.random.split(new_key, 2)

        if is_atari:
            feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
        policy = make_policy(_unpmap(training_state.params.policy))

        data = []
        for step in range(Config.batch_size * Config.num_minibatches // Config.num_envs):
            transitions = []
            for unroll_step in range(Config.unroll_length):
                current_key, key_generate_unroll = jax.random.split(key_generate_unroll)  
                obs = env_state.obs
                if is_atari:
                    obs = feature_extractor(env_state.obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                nstate = envs.step(actions) 
                # NOTE: info transformed: Array[Dict] --> Dict[Array]
                state_extras = {'truncation': jnp.array([info['truncation'] for info in nstate.info])} 
                transition = Transition(  
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=nstate.obs,
                    extras={
                        'policy_extras': policy_extras,
                        'state_extras': state_extras
                })
                transitions.append(transition)
                env_state = nstate
            data.append(jax.tree_util.tree_map(lambda *x: np.stack(x), *transitions))
        data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

        epoch_rollout_time = time.time() - update_time_start
        update_time_start = time.time()

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        assert data.discount.shape[1:] == (Config.unroll_length,)

        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]),
                                    data)
        
        # as function 
        keys_sgd = jax.random.split(key_sgd, local_devices_to_use)
        new_training_state, metrics = learn(data=data, training_state=training_state, key_sgd=keys_sgd)
    
        # logging     
        training_state, env_state, metrics = _strip_weak_type((new_training_state, env_state, metrics))
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        training_walltime = time.time() - start_time # += epoch_update_time + epoch_rollout_time
        sps = env_step_per_training_step / (epoch_update_time + epoch_rollout_time)
        global_step += env_step_per_training_step
        
        current_step = int(_unpmap(training_state.env_steps))
        
        metrics = {
            'training/total_steps': current_step,
            'training/updates': training_step,
            'training/sps': np.round(sps, 3),
            'training/walltime': np.round(training_walltime, 3),
            'training/rollout_time': np.round(epoch_rollout_time, 3),
            'training/update_time': np.round(epoch_update_time, 3),
            **{f'training/{name}': float(value) for name, value in metrics.items()}
        }

        logging.info(metrics)

        # run eval
        if process_id == 0 and Config.eval_env and training_step % Config.eval_every == 0:
            eval_start_time = time.time()
            eval_steps = 0
            if is_atari:
                feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
            policy_params = _unpmap(training_state.params.policy)
            policy = make_policy(policy_params, deterministic=Config.deterministic_eval)
            while True: 
                eval_steps += 1
                
                # run eval episode & record scores + lengths
                current_key, eval_key = jax.random.split(eval_key)
                obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
                if is_atari:
                    obs = feature_extractor(env_state.obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                eval_state = eval_env.step(actions) 
                if len(eval_env.returns) >= Config.num_eval_episodes:
                    eval_returns, eval_ep_lengths = eval_env.evaluate()
                    break
            eval_state = eval_env.reset()
            eval_time = time.time() - eval_start_time
            # compute mean + std & record
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/num_steps': eval_steps,
                'eval/mean_score': np.round(np.mean(eval_returns), 3),
                'eval/std_score': np.round(np.std(eval_returns), 3),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
                'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
                'eval/eval_time': eval_time,
            }
            logging.info(eval_metrics)
            scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), metrics['training/approx_kl']))
        
    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    # final eval
    if process_id == 0 and Config.eval_env:
        eval_steps = 0
        if is_atari:
            feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
        policy_params = _unpmap(training_state.params.policy)
        policy = make_policy(policy_params, deterministic=True)
        while True: 
            eval_steps += 1
            
            # run eval episode & record scores + lengths
            current_key, eval_key = jax.random.split(eval_key)
            obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
            if is_atari:
                obs = feature_extractor(env_state.obs)
            actions, policy_extras = policy(obs, current_key)
            actions = np.asarray(actions)
            eval_state = eval_env.step(actions) 
            if len(eval_env.returns) >= Config.num_eval_episodes:
                eval_returns, eval_ep_lengths = eval_env.evaluate()
                break
        eval_state = eval_env.reset()
        # compute mean + std & record
        eval_metrics = {
            'final_eval/num_episodes': len(eval_returns),
            'final_eval/num_steps': eval_steps,
            'final_eval/mean_score': np.mean(eval_returns),
            'final_eval/std_score': np.std(eval_returns),
            'final_eval/mean_episode_length': np.mean(eval_ep_lengths),
            'final_eval/std_episode_length': np.std(eval_ep_lengths),
        }
        logging.info(eval_metrics)
        scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), None))

        # save scores 
        run_dir = os.path.join('experiments', run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        with open(os.path.join(run_dir, "scores.pkl"), "wb") as f:
            pickle.dump(scores, f)

    if Config.save_model:
        model_path = f"weights/{run_name}.params"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(Config),
                        [
                            training_state.params.policy,
                            training_state.params.value,
                            # agent_state.params.feature_extractor,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    app.run(main)