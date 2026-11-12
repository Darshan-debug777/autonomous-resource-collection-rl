
import streamlit as st
import pandas as pd

from environment import generate_map, get_resource_sequence
from agent import Agent
from q_learning import QLearningAgent
from main import train


st.set_page_config(
    page_title="Resource Collection RL",
    page_icon="🧠",
    layout="wide"
)

st.title("Autonomous Resource Collection")
st.write(
    "Compare First Fit, Q-Learning and Optimal Knapsack "
    "on the same randomly generated resource map."
)


# Train the Q-learning agent once and reuse it.
@st.cache_resource(show_spinner="Training Q-Learning agent...")
def load_trained_agent():
    learner = QLearningAgent(seed=42)
    train(learner)
    return learner


def run_first_fit(sequence):
    collector = Agent()
    selected_positions = set()

    for position, resource in sequence:
        if collector.collect(resource):
            selected_positions.add(position)

    return collector, selected_positions


def run_q_learning(sequence, learner):
    collector = Agent()
    selected_positions = set()

    for position, resource in sequence:
        remaining = collector.capacity - collector.current_weight

        state = learner.get_state(resource, remaining, position)
        can_accept = resource["weight"] <= remaining

        action = learner.choose_action(
            state,
            can_accept,
            epsilon=0
        )

        if action == 1:
            collector.collect(resource)
            selected_positions.add(position)

    return collector, selected_positions


def run_optimal(sequence, capacity=50):
    resources = [resource for position, resource in sequence]
    n = len(resources)

    # DP table: rows = resources, columns = capacities
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight = resources[i - 1]["weight"]
        value = resources[i - 1]["value"]

        for current_capacity in range(capacity + 1):
            dp[i][current_capacity] = dp[i - 1][current_capacity]

            if weight <= current_capacity:
                candidate = (
                    dp[i - 1][current_capacity - weight] + value
                )
                dp[i][current_capacity] = max(
                    dp[i][current_capacity],
                    candidate
                )

    # Reconstruct which resources were selected.
    selected_indices = set()
    remaining_capacity = capacity

    for i in range(n, 0, -1):
        if dp[i][remaining_capacity] != dp[i - 1][remaining_capacity]:
            selected_indices.add(i - 1)
            remaining_capacity -= resources[i - 1]["weight"]

    selected_positions = {
        sequence[i][0] for i in selected_indices
    }

    total_weight = sum(
        resources[i]["weight"] for i in selected_indices
    )

    total_value = sum(
        resources[i]["value"] for i in selected_indices
    )

    return total_weight, total_value, selected_positions


def make_grid(grid, selected_positions):
    display = []

    for row in range(10):
        display_row = []

        for col in range(10):
            position = row * 10 + col
            resource = grid[row][col]

            if resource is None:
                cell = "·"
            else:
                weight = resource["weight"]
                value = resource["value"]

                if position in selected_positions:
                    cell = f"W{weight} V{value} ✓"
                else:
                    cell = f"W{weight} V{value} —"

            display_row.append(cell)

        display.append(display_row)

    return pd.DataFrame(
        display,
        index=range(10),
        columns=range(10)
    )


# Input
seed = st.number_input(
    "Enter map seed",
    min_value=0,
    max_value=2147483647,
    value=99,
    step=1
)

if st.button("Run Simulation", type="primary"):
    learner = load_trained_agent()

    grid = generate_map(int(seed))
    sequence = get_resource_sequence(grid)

    first, first_selected = run_first_fit(sequence)
    q_agent, q_selected = run_q_learning(sequence, learner)
    optimal_weight, optimal_value, optimal_selected = run_optimal(sequence)

    st.session_state["simulation"] = {
        "seed": int(seed),
        "grid": grid,
        "sequence": sequence,
        "first": first,
        "first_selected": first_selected,
        "q_agent": q_agent,
        "q_selected": q_selected,
        "optimal_weight": optimal_weight,
        "optimal_value": optimal_value,
        "optimal_selected": optimal_selected
    }


# Results
if "simulation" in st.session_state:
    result = st.session_state["simulation"]

    st.divider()
    st.subheader(f"Results for seed: {result['seed']}")

    first = result["first"]
    q_agent = result["q_agent"]
    optimal_weight = result["optimal_weight"]
    optimal_value = result["optimal_value"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("First Fit")
        st.metric("Total value", first.total_value)
        st.metric("Weight collected", f"{first.current_weight}/50")

    with col2:
        st.subheader("Q-Learning")
        st.metric("Total value", q_agent.total_value)
        st.metric("Weight collected", f"{q_agent.current_weight}/50")

    with col3:
        st.subheader("Optimal Knapsack")
        st.metric("Total value", optimal_value)
        st.metric("Weight collected", f"{optimal_weight}/50")

    st.divider()
    st.subheader("Resource collection grids")
    st.caption(
        "W = weight | V = value | ✓ = collected | — = not collected | · = empty cell"
    )

    grid1, grid2, grid3 = st.columns(3)

    with grid1:
        st.markdown("**First Fit**")
        st.dataframe(
            make_grid(result["grid"], result["first_selected"]),
            use_container_width=True
        )

    with grid2:
        st.markdown("**Q-Learning**")
        st.dataframe(
            make_grid(result["grid"], result["q_selected"]),
            use_container_width=True
        )

    with grid3:
        st.markdown("**Optimal Knapsack**")
        st.dataframe(
            make_grid(result["grid"], result["optimal_selected"]),
            use_container_width=True
        )

    st.divider()
    st.subheader("Performance comparison")

    comparison = pd.DataFrame({
        "Method": ["First Fit", "Q-Learning", "Optimal"],
        "Total value": [
            first.total_value,
            q_agent.total_value,
            optimal_value
        ]
    }).set_index("Method")

    st.bar_chart(comparison)

    if optimal_value > 0:
        gap = ((optimal_value - q_agent.total_value) / optimal_value) * 100
        st.write(f"**Q-Learning optimality gap:** {gap:.2f}%")
