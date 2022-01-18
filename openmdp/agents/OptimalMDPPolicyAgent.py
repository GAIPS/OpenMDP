import numpy as np
from yaaf import Timestep

from yaaf.agents import Agent
from openmdp import MarkovDecisionProcess


class OptimalMDPPolicyAgent(Agent):

    def __init__(self, mdp: MarkovDecisionProcess, name="Optimal MDP Policy"):
        self._mdp = mdp
        self._policy = mdp.optimal_policy
        super().__init__(name)

    def policy(self, state: np.ndarray):
        x = self._mdp.state_index(state)
        pi = self._policy[x]
        return pi

    def _reinforce(self, timestep: Timestep):
        pass
