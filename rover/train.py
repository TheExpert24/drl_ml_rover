#trained drl neural network that controls a simulated rover
from stable_baselines3 import PPO
from environment import RoverEnv

env = RoverEnv()
#proximal policy optimization
model = PPO(
    #multi-layer perception- neural network
    "MlpPolicy",
    env,
    verbose=1,
    #standard hyperparameter
    learning_rate=0.0003,
    #2048 steps in environment before update
    n_steps=2048,
    #processes the experience into smaller groups when ppo runs. 
    batch_size=64,
    #how much the brain values future rewards
    gamma=0.99
)
#agent interacts with the environment 100,000 times
model.learn(total_timesteps=100000)
model.save("rover_brain")

env.close()