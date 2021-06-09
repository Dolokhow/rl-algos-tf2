import gym
from rltf2.utils.training import Trainer
from rltf2.utils.env import GymInterface, GymRenderer, RenderDummy
from rltf2.agents.sac import SAC
from rltf2.agents.diayn import DIAYN

CONFIG_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/docs/config.yaml'
# Pendulum-v0, Hopper-v2, Ant-v2, HalfCheetah-v2
ENV_NAME = "InvertedPendulum-v2"
CUSTOM_RENDER = False
USE_DIYAN = True
NUM_OPTIONS = 50

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env_interface = GymInterface(env=env)
    obs_shape = env_interface.get_obs_shape()
    act_shape = env_interface.get_action_shape()
    max_action = env_interface.get_action_limit()

    if CUSTOM_RENDER is None:
        renderer = RenderDummy()
    else:
        renderer = GymRenderer(custom_render=CUSTOM_RENDER)

    # Taken from original DIYAN: mujoco_all_diayn.py 176-185
    # obs_space = env.spec.observation_space
    # assert isinstance(obs_space, spaces.Box)
    # low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    # high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    # aug_obs_space = spaces.Box(low=low, high=high)
    # aug_env_spec = EnvSpec(aug_obs_space, env.spec.action_space)
    if not USE_DIYAN:
        policy = SAC(
            action_shape=act_shape,
            obs_shape=obs_shape,
            max_action=max_action,
            lr=3e-4,
            actor_units=[256, 256],
            critic_units=[256, 256],
            smooth_fact=0.005,
            temp=0.2,
            discount=0.99
        )
        dir_name = 'SAC'
    else:
        policy = DIAYN(
            action_shape=env.action_space.high.size,
            obs_shape=env.observation_space.shape,
            max_action=env.action_space.high[0],
            num_options=NUM_OPTIONS,
            lr=3e-4,
            actor_units=[300, 300],
            critic_units=[300, 300],
            discriminator_units=[300, 300],
            smooth_fact=0.005,
            temp=0.1,
            discount=0.99
        )
        dir_name = 'DIAYN'

    trainer = Trainer(
        env=env_interface,
        agent=policy,
        renderer=renderer,
        root_dir='/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/new_method/' + dir_name,
        train_param_yml_path=CONFIG_PATH,
        eval_env=None,
        name=dir_name + '_' + ENV_NAME + '-cpprb'
    )
    trainer.train()
