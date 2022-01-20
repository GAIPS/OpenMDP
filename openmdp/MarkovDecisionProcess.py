from typing import Sequence, Optional

import yaaf
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from gym.envs.registration import EnvSpec

class MarkovDecisionProcess(Env):

    def __init__(self, name: str,
                 states: Sequence[np.ndarray],
                 num_actions: int,
                 transition_probabilities: np.ndarray,
                 rewards: np.ndarray,
                 discount_factor: float,
                 initial_state_distribution: np.ndarray,
                 state_meanings: Optional[Sequence[str]] = None,
                 action_meanings: Optional[Sequence[str]] = None,
                 terminal_states: Sequence[np.ndarray] = None):

        super(MarkovDecisionProcess, self).__init__()

        # MDP (S, A, P, R, gamma, miu)
        self._states = states
        self._num_states = len(states)
        self.state_meanings = state_meanings or ["<UNK>" for _ in range(self._num_states)]

        self._num_actions = num_actions
        self.action_meanings = action_meanings or ["<UNK>" for _ in range(self._num_actions)]

        self._P = transition_probabilities
        self._R = rewards
        self._discount_factor = discount_factor
        self._miu = initial_state_distribution

        # Metadata (OpenAI Gym)
        self.spec = EnvSpec(id=name)
        states_tensor = np.array(states).astype(np.float)
        self.observation_space = Box(
            low=states_tensor.min(),
            high=states_tensor.max(),
            shape=self._states[0].shape,
            dtype=states_tensor.dtype)
        self.action_space = Discrete(self._num_actions)
        self.reward_range = (rewards.min(), rewards.max())
        self.metadata = {}

        self._state = self.reset()
        self._terminal_states = terminal_states if terminal_states is not None else []

    # ########## #
    # OpenAI Gym #
    # ########## #

    def reset(self):
        x = np.random.choice(range(self.num_states), p=self._miu)
        initial_state = self._states[x]
        self._state = initial_state
        return initial_state

    def step(self, action):
        next_state = self.transition(self.state, action)
        reward = self.reward(self.state, action)
        is_terminal = self.is_terminal(next_state)
        self._state = next_state
        return next_state, reward, is_terminal, {}

    # ### #
    # MDP #
    # ### #

    def transition(self, state, action):
        x = self.state_index(state)
        y = np.random.choice(self.num_states, p=self.P[action, x])
        next_state = self.states[y]
        return next_state

    def reward(self, state, action):
        x = self.state_index(state) if not isinstance(state, int) else state
        return self.R[x, action]

    def is_terminal(self, state):
        try:
            return yaaf.ndarray_index_from(state, self._terminal_states)
        except ValueError:
            return False

    @property
    def optimal_policy(self, method="policy iteration", **kwargs) -> np.ndarray:
        if not hasattr(self, "_pi"):
            greedy_q_value_tolerance = kwargs["greedy_q_value_tolerance"] if "greedy_q_value_tolerance" in kwargs else 10e-10
            if method == "policy iteration":
                self._pi = self.policy_iteration(greedy_q_value_tolerance)
            elif method == "value iteration":
                min_error = kwargs["min_error"] if "min_error" in kwargs else 10e-8
                V = self.value_iteration(min_error)
                Q = self.q_values(V)
                self._pi = self.extract_policy(Q, greedy_q_value_tolerance)
            else:
                raise ValueError(f"Invalid solution method for {self.spec.id} '{method}'")
        return self._pi

    @property
    def optimal_q_values(self) -> np.ndarray:
        if not hasattr(self, "_q_values"):
            V = self.optimal_values
            self._q_values = self.q_values(V)
        return self._q_values

    @property
    def optimal_values(self) -> np.ndarray:
        if not hasattr(self, "_values"):
            self._values = self.value_iteration()
        return self._values

    def value_iteration(self, min_error=1e-8):
        V = np.zeros(self.num_states)
        converged = False
        while not converged:
            Q = self.q_values(V)
            V_next = np.max(Q, axis=1)
            converged = np.linalg.norm(V - V_next) <= min_error
            V = V_next
        return V

    def policy_iteration(self, greedy_q_value_tolerance=1e-10):
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        converged = False
        while not converged:
            Q = self.policy_q_values(policy)
            next_policy = self.extract_policy(Q, greedy_q_value_tolerance)
            converged = (policy == next_policy).all()
            policy = next_policy
        return policy

    def policy_values(self, policy: np.ndarray):
        if policy.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid policy shape {policy.shape}. Policies for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        R_pi = (policy * self.R).sum(axis=1)
        P_pi = np.zeros((self.num_states, self.num_states))
        for a in range(self.num_actions): P_pi += policy[:, a].reshape(-1, 1) * self.P[a]
        V_pi = np.linalg.inv(np.eye(self.num_states) - self.gamma * P_pi).dot(R_pi)
        return V_pi

    def q_values(self, values: np.ndarray):
        if values.shape != (self.num_states,):
            raise ValueError(f"Invalid values shape {values.shape}. Values for {self.spec.id} should have shape {(self.num_states,)}")
        values_as_column = values.reshape(-1, 1)
        Q = np.array([self.R[:, a].reshape(-1, 1) + self.gamma * self.P[a].dot(values_as_column) for a in range(self.num_actions)])[:, :, -1].T
        return Q

    def policy_q_values(self, policy: np.ndarray):
        if policy.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid policy shape {policy.shape}. Policies for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        V_pi = self.policy_values(policy)
        Q_pi = self.q_values(V_pi)
        return Q_pi

    def extract_policy(self, q_values: np.ndarray, greedy_q_value_tolerance=10e-10) -> np.ndarray:
        if q_values.shape != (self.num_states, self.num_actions):
            raise ValueError(f"Invalid q-values shape {q_values.shape}. Q-Values for {self.spec.id} should have shape {(self.num_states, self.num_actions)}")
        Q_greedy = np.isclose(q_values, q_values.max(axis=1, keepdims=True), atol=greedy_q_value_tolerance, rtol=greedy_q_value_tolerance).astype(int)
        policy = Q_greedy / Q_greedy.sum(axis=1, keepdims=True)
        return policy

    # ########## #
    # Properties #
    # ########## #

    @property
    def state(self):
        return self._state

    @property
    def states(self):
        return self._states

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def transition_probabilities(self):
        """Returns the Transition Probabilities P (array w/ shape X, X)"""
        return self._P

    @property
    def P(self):
        """Alias for self.transition_probabilities"""
        return self.transition_probabilities

    @property
    def rewards(self):
        """Returns the Rewards R (array w/ shape X, A)"""
        return self._R

    @property
    def R(self):
        """Alias for self.rewards"""
        return self.rewards

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def gamma(self):
        """Alias for self.discount_factor"""
        return self._discount_factor

    @property
    def initial_state_distribution(self):
        return self._miu

    @property
    def miu(self):
        return self._miu

    @property
    def terminal_states(self):
        return self._terminal_states

    # ######### #
    # Auxiliary #
    # ######### #

    def state_index(self, state=None):
        """
            Returns the index of a given state in the state space.
            If the state is unspecified (None), returns the index of the current state st.
        """
        return self.state_index(self.state) if state is None else self.state_index_from(self.states, state)

    @staticmethod
    def state_index_from(states, state):
        """Returns the index of a state (array) in a list of states"""
        return yaaf.ndarray_index_from(states, state)

    # ########### #
    # Persistence #
    # ########### #

    def save(self, directory):
        yaaf.mkdir(f"{directory}/{self.spec.id}")
        mdp = {
            "name": self.spec.id,
            "X": self.states,
            "A": self.action_meanings,
            "P": self.P,
            "R": self.R,
            "gamma": self.gamma,
            "miu": self.miu
        }
        for var in mdp:
            np.save(f"{directory}/{self.spec.id}/{var}", mdp[var])

    @staticmethod
    def load(directory):
        name = str(np.load(f"{directory}/name.npy"))
        X = np.load(f"{directory}/X.npy")
        A = list(np.load(f"{directory}/A.npy"))
        P = np.load(f"{directory}/P.npy")
        R = np.load(f"{directory}/R.npy")
        gamma = float(np.load(f"{directory}/gamma.npy"))
        miu = np.load(f"{directory}/miu.npy")
        return MarkovDecisionProcess(name, X, len(A), P, R, gamma, miu, action_meanings=A)
