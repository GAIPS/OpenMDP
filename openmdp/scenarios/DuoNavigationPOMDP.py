import yaaf
import numpy as np
from openmdp.PartiallyObservableMarkovDecisionProcess import PartiallyObservableMarkovDecisionProcess
from openmdp.scenarios.DuoNavigationMDP import draw, generate_mdp, random_goal, ABSORBENT


class DuoNavigationPOMDP(PartiallyObservableMarkovDecisionProcess):
    
    """
    A partial observable scenario where two agents (one controllable, one not)
    must reach two destinations.

    The first and controllable agent may only observe walls and teammate when standing next to them (with a probability
    of failing to observe).
    """
    
    def __init__(self, world_size=(3, 3), goals="random", noise=0.20, discount_factor: float = 0.95, name="DuoNavigationPOMDP-v1"):

        goals = goals if goals != "random" else random_goal(world_size[0])
        states, action_meanings, transition_probabilities, rewards_matrix, miu = generate_mdp(world_size, noise, goals)
        observations = self._generate_observations()
        observation_probabilities = self._generate_observation_probabilities(world_size, states, action_meanings, observations, noise)

        super(DuoNavigationPOMDP, self).__init__(name,
                         states, len(action_meanings), observations,
                         transition_probabilities, observation_probabilities,
                         rewards_matrix, discount_factor, miu,
                         action_meanings=action_meanings)

        self._world_size = world_size
        self._goals = goals

    def _generate_observations(self):
        observations = [
            np.array((up, down, left, right))
            for up in range(3)
            for down in range(3)
            for left in range(3)
            for right in range(3)
        ]
        return observations

    def _generate_observation_probabilities(self, world_size, states, action_meanings, observations, noise):

        def get_possible_observations(next_state):

            # If we are in the absorbent state, see nothing in the four directions
            if tuple(next_state) == tuple(ABSORBENT):
                observations_per_direction = [
                    {0: 1.0},
                    {0: 1.0},
                    {0: 1.0},
                    {0: 1.0}
                ]
            else:
                num_rows, num_columns = world_size
                agent_cell, teammate_cell = next_state[:2], next_state[2:4]
                (x1, y1), (x2, y2) = agent_cell, teammate_cell

                # Walls
                walls = [
                    y1 == 0,                # Wall up
                    y1 == num_rows - 1,     # Wall down
                    x1 == 0,                # Wall left
                    x1 == num_columns - 1   # Wall right
                ]

                # Agent
                adjacent_agent = [
                    y1 == y2 + 1 and x1 == x2,  # Agent up
                    y1 == y2 - 1 and x1 == x2,  # Agent down
                    y1 == y2 and x1 == x2 + 1,  # Agent left
                    y1 == y2 and x1 == x2 - 1   # Agent right
                ]
                assert np.array(adjacent_agent).sum() <= 1.0, "Agent seems to be detected in more than one direction"
                teammate_direction = adjacent_agent.index(True) if True in adjacent_agent else None

                observations_per_direction = []
                for direction, there_is_wall in enumerate(walls):

                    there_is_teammate = teammate_direction is not None and direction == teammate_direction

                    if there_is_wall:
                        possibilities = {
                            0: noise / 2,      # Detect nothing (error)
                            1: 1.0 - noise,    # Detect wall
                            2: noise / 2,      # Detect teammate as wall (error)
                        }

                    elif there_is_teammate:
                        possibilities = {
                            0: noise / 2,      # Detect nothing (error)
                            1: noise / 2,      # Detect wall instead of teammate (error)
                            2: 1.0 - noise     # Detect teammate
                        }
                    else:   # There's nothing
                        possibilities = {0: 1.0}    # Detect nothing

                    observations_per_direction.append(possibilities)

            return observations_per_direction

        num_actions = len(action_meanings)
        num_states = len(states)
        num_observations = len(observations)

        O = np.zeros((num_actions, num_states, num_observations))
        for a in range(num_actions):
            for y in range(num_states):
                possible_observations_per_direction = get_possible_observations(states[y])
                for up_flag, up_prob in possible_observations_per_direction[0].items():
                    for down_flag, down_prob in possible_observations_per_direction[1].items():
                        for left_flag, left_prob in possible_observations_per_direction[2].items():
                            for right_flag, right_prob in possible_observations_per_direction[3].items():
                                probability = up_prob * down_prob * left_prob * right_prob
                                observation = np.array([up_flag, down_flag, left_flag, right_flag])
                                z = yaaf.ndarray_index_from(observations, observation)
                                O[a, y, z] = probability
        return O

    def render(self, mode='human'):
        draw(self.state, self._world_size, self._goals)
