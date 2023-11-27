# On-Policy Policy Gradient Algorithms in JAX

This Deep Reinforcement Learning repository contains the most prominent **On-Policy Policy Gradient** Algorithms. 
All algorithms are implemented in **JAX**. Our implementations are based on [Brax's](https://github.com/google/brax) implementation of PPO. We use [Brax's](https://github.com/google/brax) logic for policy networks and distributions and [Stable Baselines3's](https://github.com/DLR-RM/stable-baselines3) environment infrastructure to create batched environments. Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), we 
provide all algorithm logic including hyperparameters in a single file. However, for efficiency we have joint files for creating networks and distributions.


## Algorithms

We implemented the following algorithms in JAX:
* [REINFORCE](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
* [Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783)
* [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
* [V-MPO*](https://arxiv.org/abs/1909.12238)

You can read more about these algorithms in our upcoming comprehensive overview of Policy Gradient Algorithms.

*on-policy variant of [Maximum a Posteriori Policy Optimization (MPO)](https://arxiv.org/abs/1806.06920)


## Benchmark Results

We report the performance of our implementations on common MuJoCo enviroments (v4), interfaced through [Gymnasium](https://gymnasium.farama.org).

|![](/images/perf_plot_HalfCheetah.png)             |  ![](/images/perf_plot_Ant.png)|
:-------------------------:|:-------------------------:
|![](/images/perf_plot_Humanoid.png)  |  ![](/images/perf_plot_Hopper.png)|



## Get started

Prerequisites:
* Tested with Python ==3.11.6
* See requirements.txt for further dependencies (Note that that file bloated, not all libraries are actually needed.).

To run the algorithms locally, simply run the respective python file:

```bash
python ppo.py
```


## Citing CleanRL

**TBA**

If you use this repository in your work or find it useful, please cite our upcoming paper.

