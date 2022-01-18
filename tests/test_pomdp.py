from unittest import TestCase, main

from yaaf.evaluation import Metric
from yaaf.execution import TimestepRunner

from openmdp.agents import MLSAgent, QMDPAgent
from openmdp.scenarios.DuoNavigationPOMDP import DuoNavigationPOMDP

class AccumulatedRewardMetric(Metric):

    def __init__(self):
        super().__init__("Accumulated Reward Metric")
        self._accumulator = 0.0

    def __call__(self, timestep):
        self._accumulator += timestep.reward

    def reset(self):
        self._accumulator = 0.0

    def result(self):
        return self._accumulator

class POMDPTests(TestCase):

    @staticmethod
    def validate_pomdp(agent, pomdp, horizon, min_reward):
        metric = AccumulatedRewardMetric()
        runner = TimestepRunner(horizon, agent, pomdp, [metric])
        runner.run()
        assert metric.result() > min_reward, f"POMDP was not solvable when partially observable (reward={metric.result()}"

    def test_mls_agent(self):
        pomdp = DuoNavigationPOMDP()
        agent = MLSAgent(pomdp)
        self.validate_pomdp(agent, pomdp, horizon=75, min_reward=0.0)

    def test_qmdp_agent(self):
        pomdp = DuoNavigationPOMDP()
        agent = QMDPAgent(pomdp)
        self.validate_pomdp(agent, pomdp, horizon=75, min_reward=0.0)

if __name__ == '__main__':
    main()
