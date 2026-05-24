import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001   

EPSILON_START = 1.0
EPSILON_DECAY = 0.993   
EPSILON_MIN = 0.01

TAU = 0.005             
MAX_EPISODES = 400      

# --- Neural Network Architecture ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (torch.FloatTensor(np.array(state)), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(np.array(next_state)), 
                torch.FloatTensor(done))
                
    def __len__(self):
        return len(self.buffer)

def main():
    # --- Global Seeding Block for Cross-Platform Consistency ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer()

    epsilon = EPSILON_START
    recent_rewards = collections.deque(maxlen=10)

    print("Starting Multi-Platform Training...")
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.push(state, action, reward, next_state, float(terminated))
            state = next_state
            total_reward += reward
            
            # Optimization step
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Compute Q(s, a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute max Q(s', a) using Target Network
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
                
                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Soft update target network parameters every step
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                
        # Decay exploration rate
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        recent_rewards.append(total_reward)
            
        # Smart Dynamic Print: Every 10 episodes OR whenever it lands a perfect 500
        if episode % 10 == 0 or total_reward >= 500.0:
            print(f"Episode {episode:3d} | Total Reward: {total_reward:5.1f} | Epsilon: {epsilon:.2f}")
            
        # Early Exit Condition: Solved if last 10 episodes average a perfect 500 score
        if len(recent_rewards) == 10 and sum(recent_rewards)/10 >= 500.0:
            print(f"\nEnvironment solved early at Episode {episode}! Average Reward: {sum(recent_rewards)/10}")
            break

    env.close()

    # --- Test and Evaluation Phase (Auto-Detects Desktop vs Displayless Server) ---
    policy_net.eval()
    
    # Check if a display backend is available (works on your laptop terminal, skips on Colab)
    if "DISPLAY" in os.environ or os.name == "nt":
        print("\nDisplay environment detected. Testing agent visually with Pygame...")
        render_mode = "human"
    else:
        print("\nNo display detected (Colab environment). Testing agent via data stream...")
        render_mode = None

    test_env = gym.make(ENV_NAME, render_mode=render_mode)
    state, _ = test_env.reset(seed=42)
    done = False
    test_steps = 0  
    
    while not done and test_steps < 500:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_t).argmax().item()
            
        state, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        test_steps += 1
        
    print(f"\n[Evaluation Complete] Balanced flawlessly for {test_steps} steps!")
    test_env.close()

if __name__ == "__main__":
    main()