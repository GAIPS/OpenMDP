from openmdp.PartiallyObservableMarkovDecisionProcess import PartiallyObservableMarkovDecisionProcess
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.policies import greedy_policy


class MLSAgent(Agent):

    def __init__(self, pomdp: PartiallyObservableMarkovDecisionProcess):
        super(MLSAgent, self).__init__("MLS")
        self._pomdp = pomdp
        self._q_values = pomdp.optimal_q_values
        self.reset()

    def reset(self):
        self._belief = self._pomdp.miu

    def policy(self, observation=None):
        most_likely_state = self._belief.argmax()
        q_values = self._q_values[most_likely_state]
        return greedy_policy(q_values)

    def _reinforce(self, timestep: Timestep):
        _, action, _, next_obs, _, _ = timestep
        self._belief = self._pomdp.belief_update(self._belief, action, next_obs)