
import random


GRID_SIZE = 10
RESOURCE_PROBABILITY = 0.25


def generate_map(seed):

    rng = random.Random(seed)

    grid = [
        [None for _ in range(GRID_SIZE)]
        for _ in range(GRID_SIZE)
    ]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            if row == 0 and col == 0:
                continue

            if rng.random() < RESOURCE_PROBABILITY:

                grid[row][col] = {
                    "weight": rng.randint(1, 10),
                    "value": rng.randint(1, 20)
                }

    return grid


def get_resource_sequence(grid):

    resources = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            resource = grid[row][col]

            if resource is not None:

                position = row * GRID_SIZE + col

                resources.append((position, resource))

    return resources


if __name__ == "__main__":

    grid = generate_map(99)

    resources = get_resource_sequence(grid)

    print("Generated resources:", len(resources))

    for position, resource in resources:
        print(position, resource)
