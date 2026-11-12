
from agent import Agent


def first_fit(resources, capacity=50):

    agent = Agent(capacity)

    for resource in resources:

        agent.collect(resource)

    return agent


def knapsack_optimal(resources, capacity=50):

    dp = [0] * (capacity + 1)

    for resource in resources:

        weight = resource["weight"]
        value = resource["value"]

        for current_weight in range(
            capacity,
            weight - 1,
            -1
        ):

            dp[current_weight] = max(
                dp[current_weight],
                dp[current_weight - weight] + value
            )

    return dp[capacity]
