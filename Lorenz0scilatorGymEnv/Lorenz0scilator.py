import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LorenzOscillatorDiscreteEnv(gym.Env):
    def __init__(self):
        super(LorenzOscillatorDiscreteEnv, self).__init__()
        
        self.sigma = 10
        self.rho = 28
        self.beta = 8/3
        self.dt = 0.01
        self.max_steps = 3000
        self.control_interval = 50
        self.init_state = np.array([0.0, 1.0, 1.05])
        
        self.action_space = spaces.Discrete(3)  # actions: -1, 0, 1
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

    def reset(self):
        self.state = self.init_state
        self.steps = 0
        self.last_x_sign = np.sign(self.state[0])
        return self.state

    def lorenz_deriv(self, state, t, u):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y + u
        dz = x * y - self.beta * z
        return [dx, dy, dz]

    def step(self, action):
        u = action - 1  # map action 0, 1, 2 to control -1, 0, 1
        t = np.linspace(0, self.dt, 2)
        self.state = odeint(self.lorenz_deriv, self.state, t, args=(u,))[1]
        self.steps += 1

        done = self.steps >= self.max_steps

        current_x_sign = np.sign(self.state[0])
        reward = 1 if current_x_sign != self.last_x_sign else 0
        self.last_x_sign = current_x_sign

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Register the environment with gym
gym.envs.registration.register(
    id='LorenzOscillatorDiscrete-v0',
    entry_point='__main__:LorenzOscillatorDiscreteEnv',
    max_episode_steps=3000,
)

# Training the RL agent
def train_agent():
    env_id = 'LorenzOscillatorDiscrete-v0'
    env = make_vec_env(env_id, n_envs=1)

    # Create the PPO agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_lorenz")
    return model

# Evaluate the RL Agent
def evaluate_agent(model):
    env_id = 'LorenzOscillatorDiscrete-v0'
    env = gym.make(env_id)
    obs = env.reset()
    states = []
    rewards = []

    # Run an episode
    for _ in range(env.max_steps):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        states.append(obs)
        rewards.append(reward)
        if done:
            break

    states = np.array(states)

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.5)
    ax.set_title("Lorenz Attractor - Controlled by PPO")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Animation
    def update(num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])
        return line,

    data = np.array(states).T
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    line, = ax2.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], lw=0.5)

    ani = FuncAnimation(fig2, update, frames=env.max_steps, fargs=(data, line), interval=1, blit=False)
    plt.show()

if __name__ == '__main__':
    # Train the agent
    model = train_agent()
    
    # Evaluate the trained agent
    evaluate_agent(model)
