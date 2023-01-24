import cv2
import numpy as np
from tqdm import tqdm

# Load the image of the labyrinth
img = cv2.imread("df_maze.png")
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to binary
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find the start and goal coordinates
start = None
for y in range(binary.shape[0]):
    for x in range(binary.shape[1]):
        if binary[y, x] == 255:
            if start is None:
                start = (y, x)
                break

end = None
for y in reversed(range(binary.shape[0])):
    for x in reversed(range(binary.shape[1])):
        if binary[y, x] == 255:
            if end is None:
                end = (y, x)
                break

# Define the possible actions
actions = ["up", "down", "left", "right"]

# Define the Q-table
q_table = np.zeros((binary.shape[0], binary.shape[1], len(actions)))

# Define the learning rate
alpha = 0.8

# Define the discount factor
gamma = 0.95

# Define the exploration rate
epsilon = 0.1

# Define the maximum number of episodes
episodes = 100  # 10000 takes too much time

# Train the Q-learning algorithm
for episode in tqdm(range(episodes)):
    # Set the initial state
    state = start

    # Loop until the agent reaches the goal
    while (abs(state[0]) < len(binary[0])) and (abs(state[1]) < len(binary[1])):
        # Select the action with the highest Q-value
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]

        # Take the action and observe the new state and reward
        if action == "up":
            new_state = (state[0] - 1, state[1])
            reward = -1 if binary[new_state[0], new_state[1]] == 0 else 0
        elif action == "down":
            new_state = (state[0] + 1, state[1])
            reward = -1 if binary[new_state[0], new_state[1]] == 0 else 0 
        elif action == "left": 
            new_state = (state[0], state[1] - 1)
            reward = -1 if binary[new_state[0], new_state[1]] == 0 else 0 
        elif action == "right": 
            new_state = (state[0], state[1] + 1)
            reward = -1 if binary[new_state[0], new_state[1]] == 0 else 0
    
        # Update the Q-value
        q_table[state[0], state[1], actions.index(action)] = q_table[state[0], state[1], actions.index(action)] + alpha * (reward + gamma * np.max(q_table[new_state[0], new_state[1]]) - q_table[state[0], state[1], actions.index(action)])

        # Update the state
        state = new_state

        # Check if the goal is reached
        if state == goal:
            break

# Print the Q-table
print(q_table)
