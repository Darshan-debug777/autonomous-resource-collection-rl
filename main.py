
from environment import generate_map, get_resource_sequence
from agent import Agent
from q_learning import QLearningAgent
from experiments import evaluate


EPISODES = 50000
TEST_MAPS = 1000


def train(learner):

    print("Starting Q-learning training...")

    for episode in range(EPISODES):

        grid = generate_map(episode)

        sequence = get_resource_sequence(grid)

        collector = Agent()

        # Exploration decreases gradually
        epsilon = max(
            0.05,
            1.0 - 0.95 * episode / (EPISODES - 1)
        )

        for i, (position, resource) in enumerate(sequence):

            remaining = (
                collector.capacity - collector.current_weight
            )

            state = learner.get_state(
                resource,
                remaining,
                position
            )

            can_accept = (
                resource["weight"] <= remaining
            )

            action = learner.choose_action(
                state,
                can_accept,
                epsilon
            )

            reward = 0

            if action == 1:

                collector.collect(resource)

                reward = resource["value"]

            next_state = None
            next_can_accept = False

            if i + 1 < len(sequence):

                next_position, next_resource = sequence[i + 1]

                next_remaining = (
                    collector.capacity - collector.current_weight
                )

                next_state = learner.get_state(
                    next_resource,
                    next_remaining,
                    next_position
                )

                next_can_accept = (
                    next_resource["weight"] <= next_remaining
                )

            learner.update(
                state,
                action,
                reward,
                next_state,
                next_can_accept
            )

        if (episode + 1) % 10000 == 0:

            print(
                "Completed episodes:",
                episode + 1,
                "| Learned states:",
                len(learner.q_table)
            )

    print("\nTraining completed.")


if __name__ == "__main__":

    learner = QLearningAgent(seed=42)

    train(learner)

    print("\nStarting evaluation...")

    evaluate(learner, TEST_MAPS)
