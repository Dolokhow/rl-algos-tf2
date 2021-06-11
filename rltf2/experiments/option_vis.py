import gym
import json
from rltf2.utils.training import Runner
from rltf2.utils.metric_inspection import sorted_opt_performance
from rltf2.utils.env import GymInterface, GymRenderer
from rltf2.agents.diayn import DIAYN
from rltf2.utils.file_io import yaml_to_dict, dict_to_yaml, check_create_dir, \
    create_dir, tensorboard_structured_summaries, get_logger, configure_tf_checkpoints

STORE_DIR = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/' \
            'vids'
CKPT_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/' \
            '/weights/ckpt-501'
TRAIN_TRACKER_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/' \
                     'logs/train_tracker.json'
# Pendulum-v0, Hopper-v2, Ant-v2, HalfCheetah-v2
ENV_NAME = "Hopper-v2"

# Agent restoration parameters
NUM_OPTIONS = 50

# Options ranking based on performance will be calculated based on stored agent performance within train_tracker.json
# Indices below indicate which options to visualize as videos based on sorted options ranking (0 being the best)
RANKING_INDICES_TO_VIS = [0, 1, 2, 3, 4, 8, 9, 18, 19, 28, 29, 38, 39, 48, 49]
# Maximum step to consider when picking best options from history
MAX_STEP = 100000


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env_interface = GymInterface(env=env)
    obs_shape = env_interface.get_obs_shape()
    act_shape = env_interface.get_action_shape()
    max_action = env_interface.get_action_limit()

    renderer = GymRenderer(custom_render=True, store_dir=STORE_DIR)

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

    runner = Runner(
        env=env_interface,
        agent=policy,
        total_steps=3000,
        intervals={},
        tracker=None,
        renderer=renderer,
        max_steps=1000,
        opt_persist=True,
        logger=get_logger(name='Vis', log_dir=STORE_DIR)
    )
    runner.load_agent(ckpt_path=CKPT_PATH)

    train_tracker = json.load(open(TRAIN_TRACKER_PATH, 'r'))
    options_id_ranking_tuple_lst = sorted_opt_performance(
        options_history=train_tracker['options_history'],
        max_total_step=MAX_STEP,
        last_n_runs=2
    )
    print(0)
    # options_to_test = runner.tracker.options_ranking[:10]
    # options_to_test.extend(runner.tracker.options_ranking[-5:])


    # for option_id in options_to_test:
    #     self.eval_runner.cur_opt_id = option_id
    #     _ = self.eval_runner.run()
    #     # Could be a hard reset as well, this way it will keep track of returns and options history
    #     # through all evaluation calls, ie. it's Tracker will not be reset!
    #     # self.eval_runner.soft_reset()
    # eval_summaries = self.eval_runner.tracker.get_summaries()
    # self._store_summaries(summaries=eval_summaries)