# PBT_MARL_watered_down

# What's in this repo?
My attempt to reproduce a water down version of PBT (Population based training) for MARL (Multi-agent reinforcement learning) inspired by Algorithm 1 (PBT-MARL) on page 3 of this [paper](https://arxiv.org/pdf/1902.07151.pdf)[1].

# MAIN differences from the paper:
(1) A simple 1 VS 1 [RockPaperScissorsEnv](https://github.com/ray-project/ray/blob/57544b1ff9f97d4da9f64d25c8ea5a3d8d247ffc/rllib/examples/env/rock_paper_scissors.py) environment (adapted & modified from a toy example from ray) is used instead of the 2 VS 2 [dm_soccer](https://git.io/dm_soccer).

(2) PPO is used instead of SVG0.

(3) No reward shaping.

(4) The evolution eligibility documented in B2 on page 16 in the [paper](https://arxiv.org/pdf/1902.07151.pdf)[1] is not implemented.

(5) Probably many more...

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

# Example of what's stored in the global storage:
```
"""
{'agt_0':
    {'hyperparameters':
        {'lr': [0.0027558622259168833, 0.0022046897807335066, 0.043092949371677264, 0.03447435949734181, 0.0031554678449498097, 0.08935753981710998],
        'gamma': [0.9516804908336309, 0.9516804908336309, 0.9847936135983785, 0.9847936135983785, 0.9269904902574178, 0.9847936135983785]},

      'opponent': ['NA', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5',     'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_5', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4', 'agt_4'],

      'score': [0, -4.0, -2.0, -4.0, -1.0, -2.0, 2.0, -3.0, 1.0, -6.0, 2.0, 0.0, 1.0, 2.0, 3.0, -1.0, 1.0, -4.0, 3.0, 1.0, -1.0, 4.0, 2.0, -4.0, 1.0, 2.0, 7.0, -2.0, -2.0, -1.0, -2.0, 3.0, -2.0, -2.0, 1.0, 2.0, -1.0],

      'rating': [0.0, 0.05, 0.05, 0.1, 0.1, 0.15000000000000002, 0.05, 0.1, 0.05, 0.1,   0.05, 0.05, 0.0, -0.05, -0.1, 0.0, -0.05, 0.05, 0.0, -0.04999281374137658, 1.4372517246834249e-05, -0.04997844122412975, -0.09997125496550632, -0.04996406870688291, -0.09995688244825948, -0.14994969618963605, -0.19994250993101265, -0.14993532367238924, -0.09992813741376583, -0.04992095115514241, 8.623510348100549e-05, -0.04990657863789558, 0.00010060762072783974, 0.050107793879351256, -0.04989220612064874, -0.09988501986202533, 0.00011498013797467399],

      'step': [0]},

 'agt_1': ...
    .
    .
    .
 'agt_n': ...    
}
"""
```

# How to run the contents in this repo?
The easiest way is to run the `PBT_MARL_watered_down.ipynb` Jupyter notebook in Colab.

# Dependencies:
This is developed & tested on Colab & the following are the only packages that I explicitly `pip install`:

ray[rllib]==0.8.5

tensorflow==2.2.0

# Disclaimer:
(1) I'm not affiliated with any of the authors of the [paper](https://arxiv.org/pdf/1902.07151.pdf)[1].

# References:
[1] [EMERGENT COORDINATION THROUGH COMPETITION (Liu et al., 2019)](https://arxiv.org/pdf/1902.07151.pdf)
