

import random


REJECT = 0
ACCEPT = 1


class QLearningAgent:

    def __init__(self, seed=42):

        self.q_table = {}
        self.rng = random.Random(seed)

        self.alpha = 0.1
        self.gamma = 1.0

    def get_state(self, resource, remaining_capacity, position):

        capacity_bucket = remaining_capacity // 5

        progress_stage = min(position // 20, 4)

        return (
            capacity_bucket,
            resource["weight"],
            resource["value"],
            progress_stage
        )

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = [0.0, 0.0]

        return self.q_table[state]

    def choose_action(self, state, can_accept, epsilon=0):

        valid_actions = [REJECT]

        if can_accept:
            valid_actions.append(ACCEPT)

        # Exploration: try a random valid action
        if self.rng.random() < epsilon:

            return self.rng.choice(valid_actions)

        q_values = self.get_q_values(state)

        # Exploitation: choose the action with the
        # highest learned Q-value.
        # On a tie, prefer ACCEPT when possible.
        return max(
            valid_actions,
            key=lambda action: (q_values[action], action)
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_can_accept
    ):

        current_q = self.get_q_values(state)[action]

        if next_state is None:

            future_q = 0

        else:

            next_q_values = self.get_q_values(next_state)

            valid_next_actions = [REJECT]

            if next_can_accept:
                valid_next_actions.append(ACCEPT)

            future_q = max(
                next_q_values[action]
                for action in valid_next_actions
            )

        target = reward + self.gamma * future_q

        new_q = current_q + self.alpha * (
            target - current_q
        )

        self.q_table[state][action] = new_q
