import numpy as np
from tqdm import tqdm

# Define the labyrinth
labyrinth = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
])

# Define the possible actions
actions = ["up", "down", "left", "right"]

# Define the Q-table
q_table = np.zeros((labyrinth.shape[0], labyrinth.shape[1], len(actions)))

# Define the learning rate
alpha = 0.8
# Define the discount factor
gamma = 0.95
# Define the exploration rate
epsilon = 0.1
# Define the maximum number of episodes
episodes = 10000

# Train the Q-learning algorithm
for episode in tqdm(range(episodes)):
    # Set the initial state
    state = (0, 0)

    # Loop until the agent reaches the goal
    while (abs(state[0]) < len(labyrinth[0])) and (abs(state[1]) < len(labyrinth[1])):
        # Select the action with the highest Q-value
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]

        # Take the action and observe the new state and reward
        if action == "up":
            new_state = (state[0] - 1, state[1])
            reward = -1 if labyrinth[new_state[0], new_state[1]] == 1 else 0
        elif action == "down":
            new_state = (state[0] + 1, state[1])
            reward = -1 if labyrinth[new_state[0], new_state[1]] == 1 else 0
        elif action == "left":
            new_state = (state[0], state[1] - 1)
            reward = -1 if labyrinth[new_state[0], new_state[1]] == 1 else 0
        elif action == "right":
            new_state = (state[0], state[1] + 1)
            reward = -1 if labyrinth[new_state[0], new_state[1]] == 1 else 0

        # Update the Q-value
        q_table[state[0], state[1], actions.index(action)] = q_table[state[0], 
                                                                     state[1], 
                                                                     actions.index(action)] + alpha * (reward + gamma * np.max(q_table[new_state[0], new_state[1]]) - q_table[state[0], state[1], actions.index(action)])

        # Update the state
        state = new_state

        # Check if the goal is reached
        if state == (labyrinth.shape[0] - 1, labyrinth.shape[1] - 1):
            break

print(f'State: {state}')

# Print the Q-table
print(f'Q-table: \n {q_table}')
