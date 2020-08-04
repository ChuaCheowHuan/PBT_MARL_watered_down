"""#PBT_MARL class"""

import random
import numpy as np
import ray
from ray.rllib.utils.schedules import ConstantSchedule

class PBT_MARL:
    def __init__(self, population_size,
                 K, T_select,
                 binomial_n, inherit_prob,
                 perturb_prob, perturb_val):
        self.population_size = population_size      # num of agents to choose from
        self.K = K      # # step size of Elo rating update given one match result.
        self.T_select = T_select      # agt_j selection threshold
        # inherit variables
        self.binomial_n = binomial_n     # bernoulli is special case of binomial when n=1
        self.inherit_prob = inherit_prob     # hyperparameters are either inherited or not independently with probability 0.5
        # mutation variables
        self.perturb_prob = perturb_prob     # resample_probability
        self.perturb_val = perturb_val      # lower & upper bound for perturbation value

    def _is_eligible(self, agt_i_key):
        """
        If agt_i completed certain training steps > threshold after
        last evolution, return true.
        """
        return True

    def _is_parent(self, agt_j_key):
        """
        If agt_i completed certain training steps > threshold after
        last evolution, return true.
        """
        return True

    def _s_elo(self, rating_i, rating_j):
        return 1 / (1 + 10**((rating_j - rating_i) / 400))

    def compute_rating(self, prev_rating_i, prev_rating_j, score_i, score_j):
        s = (np.sign(score_i - score_j) + 1) / 2
        s_elo_val = self._s_elo(prev_rating_i, prev_rating_j)
        rating_i = prev_rating_i + self.K * (s - s_elo_val)
        rating_j = prev_rating_j + self.K * (s - s_elo_val)

        return rating_i, rating_j

    def _select_agt_j(self, pol_i_id, population_size, T_select):
        pol_j_id = np.random.randint(low=0, high=population_size, size=None)
        while pol_i_id == pol_j_id:
            pol_j_id = np.random.randint(low=0, high=population_size, size=None)

        agt_i_key = "agt_{}".format(str(pol_i_id))
        agt_j_key = "agt_{}".format(str(pol_j_id))

        g_helper = ray.get_actor("g_helper")
        rating_i = ray.get(g_helper.get_rating.remote(agt_i_key))
        rating_j = ray.get(g_helper.get_rating.remote(agt_j_key))

        s_elo_val = self._s_elo(rating_j, rating_i)
        #print("s_elo_val:", s_elo_val)

        if s_elo_val < T_select:
            return pol_j_id
        else:
            return None

    def _inherit(self, trainer, pol_i_id, pol_j_id):
        pol_i = "p_" + str(pol_i_id)
        pol_j = "p_" + str(pol_j_id)
        #print("{}_vs_{}".format(pol_i, pol_j))

        # cpy param_j to param_i
        self._cp_weight(trainer, pol_j, pol_i)

        # inherit hyperparam_j to hyperparam_i
        m = np.random.binomial(self.binomial_n, self.inherit_prob, size=1)[0]      # weightage to inherit from agt_i
        return self._inherit_hyperparameters(trainer, pol_j, pol_i, m)

    def _cp_weight(self, trainer, src, dest):
        """
        Copy weights of source policy to destination policy.
        """

        P0key_P1val = {}
        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            P0key_P1val[k] = v2

        trainer.set_weights({dest:P0key_P1val,
                             src:trainer.get_policy(src).get_weights()})

        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            assert (v == v2).all()

    def _inherit_hyperparameters(self, trainer, src, dest, m):
        src_pol = trainer.get_policy(src)
        #print("src_pol.config['lr']", src_pol.config["lr"])

        dest_pol = trainer.get_policy(dest)
        #print("dest_pol.config['lr']", dest_pol.config["lr"])

        dest_pol.config["lr"] = m * dest_pol.config["lr"] + (1-m) * src_pol.config["lr"]
        dest_pol.config["gamma"] = m * dest_pol.config["gamma"] + (1-m) * src_pol.config["gamma"]
        #print("src_pol.config['lr']", src_pol.config["lr"])
        #print("dest_pol.config['lr']", dest_pol.config["lr"])

        return dest_pol

    def _mutate(self, pol_i_id, pol_i):
        """
        Don't perturb gamma, just resample when applicable.
        """
        if random.random() < self.perturb_prob:     # resample
            pol_i.config["lr"] = np.random.uniform(low=0.00001, high=0.1, size=None)
            pol_i.config["gamma"] = np.random.uniform(low=0.9, high=0.999, size=None)
        elif random.random() < 0.5:     # perturb_val = 0.8
            pol_i.config["lr"] = pol_i.config["lr"] * self.perturb_val[0]
            #pol_i.config["gamma"] = pol_i.config["gamma"] * self.perturb_val[0]
        else:     # perturb_val = 1.2
            pol_i.config["lr"] = pol_i.config["lr"] * self.perturb_val[1]
            #pol_i.config["gamma"] = pol_i.config["gamma"] * self.perturb_val[1]

        # update hyperparameters in storage
        key = "agt_" + str(pol_i_id)
        g_helper = ray.get_actor("g_helper")
        ray.get(g_helper.update_hyperparameters.remote(key, pol_i.config["lr"], pol_i.config["gamma"]))

        # https://github.com/ray-project/ray/blob/051fdd8ee611e26950e104eeb9375d0d88a846d5/rllib/policy/tf_policy.py#L719
        pol_i.lr_schedule = ConstantSchedule(pol_i.config["lr"], framework=None)

    def PBT(self, trainer):
        """
        For all agents in population, if agt_i is eligible,
        select agt_j, (i != j), if agt_j is a parent,
        inherit (exploit) & mutate (explore: pertube/resample)
        """
        for i in range(self.population_size):
            pol_i_id = i
            if self._is_eligible(pol_i_id):
                #pol_j_id = self._select_agt_j(pol_i_id, self.population_size, store, self.T_select)
                pol_j_id = self._select_agt_j(pol_i_id, self.population_size, self.T_select)
                if pol_j_id is not None:
                    if self._is_parent(pol_j_id):
                        pol_i = self._inherit(trainer, pol_i_id, pol_j_id)
                        self._mutate(pol_i_id, pol_i)
