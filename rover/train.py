from stable_baselines3 import PPO
from environment import RoverEnv


def main():
    env = RoverEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    model.learn(total_timesteps=500000)
    model.save("brainzip/rover_brain_6")

    env.close()


if __name__ == "__main__":
    main()