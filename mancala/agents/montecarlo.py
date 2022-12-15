import random
from collections import defaultdict
from typing import Union
from dataclasses import dataclass

import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class MCTSRootNode:
    def __init__(self, state: BaseState, depth: int, sims: int = 50, parent=None, parent_action=None):
        self.state = state.clone()
        self.max_sims = sims
        self.depth = depth
        self.parent = parent
        self.parent_act = parent_action
        self._untried_actions = self.state.legal_actions(self.state.turn)
        self.children = []
        self._num_sims = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0

    def select(self):
        """
        Checks if all possible initial actions have been attempted.
        If not, it chooses an unattempted action, else it chooses the calculated best child.
        """
        child = self
        if not self.state._done:
            if len(self._untried_actions) != 0:
                child = self.expand()
            else:
                child = self.best_child()
        return child

    def expand(self):
        """
        Spreads the search tree into a new child that hasn't been visited.
        Sets up child data class for tracking wins.
        """
        action = self._untried_actions.pop()
        next_state = self.state.clone().proceed_action(action)
        child = MCTSChildNode(next_state, action)
        self.children.append(child)
        return child

    def simulate(self, child):
        """
        Uses cloned states to run a random simulation of a game from the child.
        Returns the player that won the simulation.
        """
        sim_state = child.state.clone()
        depth = self.depth
        while not depth == 0 and not sim_state._done:
            legal_actions = sim_state.legal_actions(sim_state.turn)
            if legal_actions is not None:
                action = legal_actions[np.random.randint(len(legal_actions))]
                sim_state = sim_state.proceed_action(action)
                depth -= 1
            else:
                sim_state = sim_state.proceed_action(None)
        return sim_state.scores[child.state.turn - 1] - sim_state.scores[child.state.turn]

    def backpropagate(self, child, reward):
        """
        Tallies the wins/losses of a child and how many attempts the child has had.
        """
        self._num_sims += 1
        child.visits += 1
        child.reward += reward

    def best_child(self, constant=0.1):
        """
        Returns list of UCB formula weights
        """
        action_weights = [((a.reward / a.visits) + constant * (np.sqrt(np.log10(self._num_sims) / a.visits))) for a in
                          self.children]
        return self.children[np.argmax(action_weights)]

    def choose_action(self):
        """
        Runs the loop and generates what probabilistically will be the best choice.
        """
        for i in range(self.max_sims):
            child = self.select()
            reward = self.simulate(child)
            self.backpropagate(child, reward)

        return self.best_child().action


@dataclass
class MCTSChildNode:
    state: BaseState
    action: int
    visits: int = 0
    reward: int = 0


class MCTSAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, id: int, depth: int = 2):
        self._seed = 42
        self.deterministic = False
        self._depth = depth
        self.set_id(id)

    def policy(self, state: BaseState) -> Union[int, None]:
        legal_actions = state.legal_actions(state.turn)
        if legal_actions is None:
            return None
        root = MCTSRootNode(state, self._depth)
        return root.choose_action()
