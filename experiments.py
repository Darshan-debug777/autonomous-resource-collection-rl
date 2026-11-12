
from environment import generate_map, get_resource_sequence
from agent import Agent
from baselines import first_fit, knapsack_optimal


def evaluate(learner, test_maps=1000):

    results = {
        "First Fit": {
            "value": 0,
            "weight": 0
        },
        "Q-Learning": {
            "value": 0,
            "weight": 0
        },
        "Optimal": {
            "value": 0
        }
    }

    total_gap_percentage = 0

    for seed in range(100000, 100000 + test_maps):

        sequence = get_resource_sequence(
            generate_map(seed)
        )

        resources = [
            resource for position, resource in sequence
        ]

        # First Fit
        first = first_fit(resources)

        # Q-Learning
        collector = Agent()

        for position, resource in sequence:

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
                epsilon=0
            )

            if action == 1:

                collector.collect(resource)

        # Offline optimal
        optimal_value = knapsack_optimal(resources)

        results["First Fit"]["value"] += first.total_value
        results["First Fit"]["weight"] += first.current_weight

        results["Q-Learning"]["value"] += collector.total_value
        results["Q-Learning"]["weight"] += collector.current_weight

        results["Optimal"]["value"] += optimal_value

        if optimal_value > 0:

            gap = (
                (optimal_value - collector.total_value)
                / optimal_value
            ) * 100

            total_gap_percentage += gap

    print("\nAVERAGE RESULTS ACROSS", test_maps, "UNSEEN MAPS")

    for name, data in results.items():

        average_value = data["value"] / test_maps

        print("\n", name)
        print("Average value:", round(average_value, 2))

        if "weight" in data:

            average_weight = data["weight"] / test_maps

            utilisation = (average_weight / 50) * 100

            print("Average weight:", round(average_weight, 2))
            print("Capacity utilisation:", round(utilisation, 2), "%")

    print(
        "\nQ-Learning average optimality gap:",
        round(total_gap_percentage / test_maps, 2),
        "%"
    )

    return results

