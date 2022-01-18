from openmdp.PartiallyObservableMarkovDecisionProcess import PartiallyObservableMarkovDecisionProcess
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.policies import greedy_policy

class QMDPAgent(Agent):

    def __init__(self, pomdp: PartiallyObservableMarkovDecisionProcess):
        super(QMDPAgent, self).__init__("QMDP")
        self._pomdp = pomdp
        self._q_values = pomdp.optimal_q_values
        self.reset()

    def reset(self):
        self._belief = self._pomdp.miu

    def policy(self, observation=None):
        q_values = self._belief.dot(self._q_values)
        return greedy_policy(q_values)

    def _reinforce(self, timestep: Timestep):
        _, action, _, next_obs, terminal, _ = timestep
        if terminal:
            self._belief = self._pomdp.miu
        else:
            self._belief = self._pomdp.belief_update(self._belief, action, next_obs)
