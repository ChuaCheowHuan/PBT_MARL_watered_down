# PBT_MARL_watered_down

# What's in this repo?
My attempt to reproduce a water down version of PBT (Population based training) for MARL (Multi-agent reinforcement learning) inspired by Algorithm 1 (PBT-MARL) on page 3 of this [paper](https://arxiv.org/pdf/1902.07151.pdf)[1].

# MAIN differences from the paper:
(1) A simple 1 VS 1 [RockPaperScissorsEnv](https://github.com/ray-project/ray/blob/57544b1ff9f97d4da9f64d25c8ea5a3d8d247ffc/rllib/examples/env/rock_paper_scissors.py) environment (adapted & modified from a toy example from ray) is used instead of the 2 VS 2 [dm_soccer](https://git.io/dm_soccer).

(2) PPO is used instead of SVG0.

(3) The evolution eligibility documented in B2 on page 16 in the [paper](https://arxiv.org/pdf/1902.07151.pdf)[1] is not implemented.

(4) Probably many more...

# What works?
(1) Policies weights can be inherited between different agents in the population.

(2) Learning rate & gamma are the only 2 hyperparameters involved for now. Both can be inherited/mutated. Learning rate can be resampled/perturbed while gamma can only be resampled.

# Simple walkthru:
Before each training iteration, the driver (in this context, the main process, this is also where the RLlib trainer resides) randomly selects a pair of agents (agt_i, agt_j, where i != j) from a population of agents. This i, j pair will take up the role of player_A & player_B respectively.

The IDs of i,j will be transmitted down to the worker processes. Each worker has 1 or more environments ([vectorized](https://rllib.readthedocs.io/en/latest/rllib-env.html#vectorized)) & does it's own rollout. When an episode is sampled (that's when a match ends), the `on_episode_end` callback will be called. That's when the ratings of a match are computed & updated to a global storage.

When enough samples are collected, training starts. Training is done using [RLlib's DDPPO](https://docs.ray.io/en/master/rllib-algorithms.html#decentralized-distributed-proximal-policy-optimization-dd-ppo) (a variant of PPO). In DDPPO, learning does not happened in the trainer. Each worker does it's own learning. However, the trainer is still involved in the weight sync.

When a training iteration completes, `on_train_results` callback will be called. That's where inheritance & mutation happens (if conditions are fulfilled).

All of the above happens during 1 single main training loop of the driver. Rinse & repeat.

Note: Global coordination between different processes is done using [detached actors](https://docs.ray.io/en/master/advanced.html#detached-actors) from ray.

# How to run the contents in this repo?
The easiest way is to run the `PBT_MARL_watered_down.ipynb` Jupyter notebook in Colab.

# Dependencies:
This is developed & tested on Colab & the following are the only packages that I explicitly `pip install`:

ray[rllib]==0.85

tensorflow==2.2.0

# Disclaimer:
(1) I'm not affiliated with any of the authors of the [paper](https://arxiv.org/pdf/1902.07151.pdf)[1].

# References:
[1] [EMERGENT COORDINATION THROUGH COMPETITION (Liu et al., 2019)](https://arxiv.org/pdf/1902.07151.pdf)
