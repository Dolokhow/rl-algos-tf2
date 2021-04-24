import gym
from rltf2.utils.experiment import GymExperiment
from rltf2.agents.sac import SAC

CONFIG_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/docs/config.yaml'
ENV_NAME = "Ant-v2"
REPLAY_BUFFER_SIZE = 1e5

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    shape = env.observation_space.shape

    policy = SAC(
        action_shape=env.action_space.high.size,
        obs_shape=env.observation_space.shape,
        max_action=env.action_space.high[0],
        lr=3e-4,
        actor_units=[256, 256],
        critic_units=[256, 256],
        smooth_fact=0.005,
        temp=0.2,
        discount=0.99
    )

    experiment = GymExperiment(
        env=env,
        agent=policy,
        store_dir='/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/SAC',
        config_path=CONFIG_PATH,
        eval_env=None,
        render_mode='human',
        name='SAC_Ant-v2-cpprb'
    )
    experiment.train()
