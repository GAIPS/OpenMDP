import argparse
from yaaf import Timestep
from openmdp.agents import QMDPAgent, MLSAgent, OptimalMDPPolicyAgent
from openmdp.scenarios import DuoNavigationPOMDP

def run(agent, pomdp, horizon, render=False):

    fully_observable = agent.name == "Optimal MDP Policy"
    state = pomdp.reset()

    if not fully_observable:
        agent.reset()

    for step in range(horizon):

        action = agent.action(state) if fully_observable else agent.action()
        next_obs, reward, _, info = pomdp.step(action)
        terminal = reward == 0.0
        next_state = pomdp.state
        timestep = Timestep(state, action, reward, next_state, terminal, info) if fully_observable else Timestep(None, action, reward, next_obs, terminal, info)
        agent.reinforcement(timestep)
        if render:
            print(f"\n##########\nTimestep {step}\n##########\n")
            pomdp.render()
            print(f"State: {state} (y={pomdp.state_index(state)})")
            print(f"Action: {pomdp.action_meanings[action]} (a={action})")
            print(f"Next state: {next_state} (y={pomdp.state_index(next_state)})")
            print(f"Next obs: {next_obs} (z={pomdp.observation_index(next_obs)})")
            print(f"Reward: {reward}")

        state = next_state
        if terminal: break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("agent", type=str, choices=["MLSAgent", "QMDPAgent", "ValueIteration"])
    parser.add_argument("--horizon", default=10, type=int)
    parser.add_argument("--render", action="store_true")

    opt = parser.parse_args()

    pomdp = DuoNavigationPOMDP()

    if opt.agent == "MLSAgent": agent = MLSAgent(pomdp)
    elif opt.agent == "ValueIteration": agent = OptimalMDPPolicyAgent(pomdp)
    elif opt.agent == "QMDPAgent": agent = QMDPAgent(pomdp)
    else: raise ValueError("Unreachable exception due to choices=[...] on argparse")

    run(agent, pomdp, opt.horizon, True)
