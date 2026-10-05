Create the virtual environment:

python3 -m .venv venv
source .venv/bin/activate

Instal Dependencies:

pip install numpy gynmnasium
pip install torch stable-baselines3

Optional. Command to test rover environment:

python -c "from environment import RoverEnv; env = RoverEnv(); obs, info = env.reset(); print('Observation:', obs); print('Action space:', env.action_space)"

Example output for train.py:

------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 65.6         | ← episode lasts no. seconds on average
|    ep_rew_mean          | 149          | ← average total reward per episode
| time/                   |              | 
|    fps                  | 1525         | ← simulator is generating no. steps/second
|    iterations           | 49           |
|    time_elapsed         | 65           |
|    total_timesteps      | 100352       | ← agent has taken about no. actions
| train/                  |              |
|    approx_kl            | 0.0022985619 | ← how much the proximal policy optimization updated
|    clip_fraction        | 0.0321       | ← % of ppo updates were clipped
|    clip_range           | 0.2          | 
|    entropy_loss         | -0.25        | ← randomness/exploratory the ppo still is.
|    explained_variance   | 0.847        | ← how well the value network returns
|    learning_rate        | 0.0003       |
|    loss                 | 1.24         |
|    n_updates            | 480          | ← number of times ppo updated its network
|    policy_gradient_loss | -0.00538     |
|    value_loss           | 3.75         |
------------------------------------------