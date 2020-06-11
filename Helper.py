"""#Helper class"""

import random
import numpy as np

import ray

@ray.remote(num_cpus=0.25, num_gpus=0)
class Helper:
    def __init__(self, population_size, policies):
        self.population_size = population_size
        self.agt_i, self.agt_j = None, None
        self.policies = policies
        self.agt_store = self._create_agt_store(population_size, policies)

    def set_pair(self):
        i, j = np.random.randint(low=0, high=self.population_size, size=2)
        while i == j:
            j = np.random.randint(low=0, high=self.population_size, size=None)

        self.agt_i = "agt_" + str(i)
        self.agt_j = "agt_" + str(j)

    def get_pair(self):
        return self.agt_i, self.agt_j

    def _create_agt_store(self, population_size, policies):
        """
        Storage for stats of agents in the population.
        """
        store = {}
        for i in range(0, population_size):
            agt_name = "agt_{}".format(str(i))
            store[agt_name] = {"hyperparameters": {"lr":[],
                                                   "gamma":[]},
                               "opponent": [],
                               "score": [],
                               "rating": [],
                               "step": []}      # Steps since last evolved.

        store = self._init_hyperparameters(store, policies)

        return store

    def _init_hyperparameters(self, store, policies):
        """
        """
        for key, val in store.items():
            _, str_i = key.split("_")
            pol_key = "p_" + str_i
            lr = policies[pol_key][3]["lr"]
            gamma = policies[pol_key][3]["gamma"]
            opponent = "NA"
            score = 0
            #rating = np.random.uniform(low=0.0, high=1.0, size=None)
            rating = 0.0
            step = 0

            store[key]["hyperparameters"]["lr"].append(lr)
            store[key]["hyperparameters"]["gamma"].append(gamma)
            store[key]["opponent"].append(opponent)
            store[key]["score"].append(score)
            store[key]["rating"].append(rating)
            store[key]["step"].append(step)

        return store

    def get_agt_store(self):
        return self.agt_store

    def update_hyperparameters(self, key, lr, gamma):
        """
        Note that the hyperparameters are not updated per episode rollout.
        They are only updated after each main training loop when applicable.
        """
        self.agt_store[key]["hyperparameters"]["lr"].append(lr)
        self.agt_store[key]["hyperparameters"]["gamma"].append(gamma)

    def update_rating(self, agt_i_key, agt_j_key, rating_i, rating_j, score_i, score_j):
        self.agt_store[agt_i_key]["opponent"].append(agt_j_key)
        self.agt_store[agt_i_key]["score"].append(score_i)
        self.agt_store[agt_i_key]["rating"].append(rating_i)

        self.agt_store[agt_j_key]["opponent"].append(agt_i_key)
        self.agt_store[agt_j_key]["score"].append(score_j)
        self.agt_store[agt_j_key]["rating"].append(rating_j)

    def get_rating(self, agt_key):
        return self.agt_store[agt_key]["rating"][-1]
