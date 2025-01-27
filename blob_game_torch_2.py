import pygame
import random
import numpy as np
import time  # Import time module for tracking elapsed time
import math  # Import math module for square root calculation
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Blob and food positions
class BlobEnv:
    def __init__(self, food_distance=10):  # food_distance is the fixed distance
        self.grid_width = WIDTH // CELL_SIZE
        self.grid_height = HEIGHT // CELL_SIZE
        self.food_distance = food_distance  # Set the fixed distance for the food
        self.reset()

    def reset(self):
        self.blob_pos = [random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)]
        self.spawn_food()  # Spawn food at a fixed distance from the blob
        self.steps = 0
        self.previous_distance = self.get_distance(self.blob_pos, self.food_pos)  # Initialize previous distance
        return self.blob_pos

    def get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def spawn_food(self):
        """Spawn food at a fixed distance from the blob."""
        # Random direction for spawning food (angle between 0 and 2π)
        angle = random.uniform(0, 2 * math.pi)
        # Calculate the offset for food position based on angle and distance
        offset_x = 2*int(self.food_distance * math.cos(angle))
        offset_y = 2*int(self.food_distance * math.sin(angle))

        # Calculate the new food position
        new_food_pos = [
            self.blob_pos[0] + offset_x,
            self.blob_pos[1] + offset_y
        ]

        # Ensure the food is within bounds of the grid
        new_food_pos[0] = max(0, min(self.grid_width - 1, new_food_pos[0]))
        new_food_pos[1] = max(0, min(self.grid_height - 1, new_food_pos[1]))

        self.food_pos = new_food_pos

    def step(self, action):
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        if action == 0 and self.blob_pos[1] > 0:
            self.blob_pos[1] -= 1
        elif action == 1 and self.blob_pos[1] < self.grid_height - 1:
            self.blob_pos[1] += 1
        elif action == 2 and self.blob_pos[0] > 0:
            self.blob_pos[0] -= 1
        elif action == 3 and self.blob_pos[0] < self.grid_width - 1:
            self.blob_pos[0] += 1

        # Calculate Manhattan distances
        x_distance = abs(self.blob_pos[0] - self.food_pos[0])
        y_distance = abs(self.blob_pos[1] - self.food_pos[1])
        new_distance = x_distance + y_distance

        # Calculate reward based on movement
        if new_distance < self.previous_distance:
            reward = 1  # +1 if closer to food
        elif new_distance == self.previous_distance:
            reward = -2  # No reward if the distance is the same
        else:
            reward = -4  # -2 if further from food

        # Update the previous distance for the next step
        self.previous_distance = new_distance

        # Check if food is reached
        done = self.blob_pos == self.food_pos
        # If reached the food, give a bigger reward
        if done:
            reward += 10  # Add 10 points for reaching the food
            self.spawn_food()  # Respawn food at a new fixed distance from the blob

        return self.blob_pos, reward, done


# Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)  # Fully connected layer
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, output_size)  # Output Q-values for each action

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment
env = BlobEnv()

# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blob Searching for Food")
clock = pygame.time.Clock()


# Hyperparameters
alpha = 0.01  # Learning rate (for the optimizer)
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
episodes = 30  # Number of episodes
input_size = 2  # We will only use the distance to the food
output_size = 4  # Four possible actions (up, down, left, right)

# Initialize neural network and optimizer
model = DQN(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()  # Mean Squared Error loss

# Track time (start time)
start_time = time.time()
episode_times = []  

# Deep Q-Learning loop
running = True
for episode in range(episodes):
    state = env.reset()
    done = False
    episode_start_time = time.time()  
    x_distance = env.food_pos[0] - state[0]
    y_distance = env.food_pos[1] - state[1]
    state_input = torch.tensor([x_distance, y_distance], dtype=torch.float32).unsqueeze(0)
    grid_width = WIDTH // CELL_SIZE
    grid_height = HEIGHT // CELL_SIZE
    state_input = torch.tensor([x_distance / grid_width, y_distance / grid_height], dtype=torch.float32).unsqueeze(0)


    while not done and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check elapsed time to quit after 15 seconds
        if time.time() - start_time > 5*60:
            running = False
        
        epsilon = max(0.01, epsilon * 0.98)  # Decay epsilon after each episode

        # Select action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  # Exploration
        else:
            with torch.no_grad():
                q_values = model(state_input)  # Predict Q-values for all actions
                action = torch.argmax(q_values).item()  # Select the action with the highest Q-value

        # Take step and get new state and reward
        next_state, reward, done = env.step(action)
       # Calculate the relative distances for the next state
        next_x_distance = env.food_pos[0] - next_state[0]
        next_y_distance = env.food_pos[1] - next_state[1]
        next_state_input = torch.tensor([next_x_distance, next_y_distance], dtype=torch.float32).unsqueeze(0)

        # Compute the target Q-value
        with torch.no_grad():
            next_q_values = model(next_state_input)  # Q-values for the next state
            target = reward + gamma * torch.max(next_q_values)

        # Compute the current Q-value
        q_values = model(state_input)
        current_q_value = q_values[0][action]

        # Compute loss and update the model
        loss = loss_fn(current_q_value, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualization
        screen.fill(WHITE)
        # Draw food
        pygame.draw.rect(
            screen,
            RED,
            pygame.Rect(env.food_pos[0] * CELL_SIZE, env.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )
        # Draw blob
        pygame.draw.rect(
            screen,
            BLUE,
            pygame.Rect(env.blob_pos[0] * CELL_SIZE, env.blob_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

        pygame.display.flip()
        clock.tick(30)  # Adjust for speed (higher = faster)

        # Set the state to the next state
        state = next_state
        state_input = next_state_input

                # If the blob reaches the food, calculate the elapsed time
        if done:
            elapsed_time = time.time() - episode_start_time
            episode_times.append(elapsed_time)

pygame.quit()

# Plot time taken to reach the reward in each episode
plt.figure(figsize=(10, 6))
plt.plot(range(len(episode_times)), episode_times, label="Time to Reach Food", color="blue")
plt.xlabel("Episode")
plt.ylabel("Time (seconds)")
plt.title("Time Taken to Reach Food Over Episodes")
plt.legend()
plt.grid(True)
plt.show()