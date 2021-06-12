import gym
import json
import numpy as np
import os
from rltf2.utils.training import Runner
from rltf2.utils.metric_inspection import sorted_opt_performance
from rltf2.utils.env import GymInterface, GymRenderer
from rltf2.agents.diayn import DIAYN
from rltf2.utils.vector_ops import int_to_str
from rltf2.utils.file_io import get_logger, find_files_by_part
from rltf2.utils.visualization_utils import merge_videos_to_matrix

STORE_DIR = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/paper_replica/DIAYN_Hopper-v2-cpprb-paper-replica/' \
            'vids'
CKPT_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/paper_replica/DIAYN_Hopper-v2-cpprb-paper-replica/' \
            '/weights/ckpt-501'
TRAIN_TRACKER_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/paper_replica/DIAYN_Hopper-v2-cpprb-paper-replica/' \
                     'logs/train_tracker.json'

# Agent restoration parameters
NUM_OPTIONS = 50
# Pendulum-v0, Hopper-v2, Ant-v2, HalfCheetah-v2
ENV_NAME = "Hopper-v2"

# Visualization params
SPECIFIC_OPTION_IDS = []
PERMUTE_SPECIFIC = False
NUM_VIDEOS_TO_MERGE = 6
N_COLS = 3

# Maximum step to consider when picking best options from history.
# CAUTION! MATCH THIS WITH CHECKPOINT USED!
MAX_STEP = 2004001


if __name__ == '__main__':
    np.random.seed(123)
    video_name_dict = {}

    if SPECIFIC_OPTION_IDS is None or len(SPECIFIC_OPTION_IDS) == 0:
        train_tracker = json.load(open(TRAIN_TRACKER_PATH, 'r'))
        options_id_ranking_tuple_lst = sorted_opt_performance(
            options_history=train_tracker['options_history'],
            max_total_step=MAX_STEP,
            metric='returns',
            last_n_runs=2
        )
        desired_options = [opt[0] for opt in options_id_ranking_tuple_lst]
    else:
        desired_options = SPECIFIC_OPTION_IDS

    options_to_be_run = []
    for opt_id in desired_options:
        identifier = str(opt_id)
        search_name = '_opt_' + identifier + '_'
        matches = find_files_by_part(folder=STORE_DIR, name_part=search_name, ext='.mp4')
        if len(matches) == 0:
            options_to_be_run.append(opt_id)
        else:
            video_name_dict[identifier] = matches[0]

    if len(options_to_be_run) > 0:

        env = gym.make(ENV_NAME)
        env_interface = GymInterface(env=env)
        obs_shape = env_interface.get_obs_shape()
        act_shape = env_interface.get_action_shape()
        max_action = env_interface.get_action_limit()

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

        renderer = GymRenderer(
            custom_render=True,
            store_dir=STORE_DIR,
            name=policy.name + '_' + int_to_str(n=MAX_STEP)
        )

        runner = Runner(
            env=env_interface,
            agent=policy,
            total_steps=300,
            intervals={},
            tracker=None,
            renderer=renderer,
            max_steps=300,
            opt_persist=True,
            logger=get_logger(name='Vis', log_dir=STORE_DIR)
        )
        runner.load_agent(ckpt_path=CKPT_PATH)

        for option in options_to_be_run:
            runner.cur_opt_id = np.int(option)
            _ = runner.run()
            matches = find_files_by_part(folder=STORE_DIR, name_part='_opt_' + str(option) + '_', ext='.mp4')
            if len(matches) == 0:
                raise ValueError('This should not happen, we should have just created a video.')
            video_name_dict[str(option)] = matches[0]

    # By now we should have visualizations for each option that is required for video merger
    # Option videos are stored in video_name_dict key: option_id, data: full video_path
    mergers = []
    if SPECIFIC_OPTION_IDS is None or len(SPECIFIC_OPTION_IDS) == 0:
        num_merged_videos = int(np.floor(NUM_OPTIONS / NUM_VIDEOS_TO_MERGE))
        for index_i in range(num_merged_videos):
            merger = []
            merger_name = ''
            for index_j in range(NUM_VIDEOS_TO_MERGE):
                opt_index = index_i + index_j*num_merged_videos
                if opt_index >= NUM_OPTIONS:
                    break

                merger.append(opt_index)
                merger_name += str(desired_options[opt_index]) + '_'
            merger = np.random.permutation(merger)
            # We extend tuple (merger_name, video_paths for merger) The first will be output video name.
            mergers.append((merger_name[:-1], [(video_name_dict[str(desired_options[opt_id])]) for opt_id in merger]))
    else:
        num_merged_videos = int(np.ceil(len(desired_options)) / NUM_VIDEOS_TO_MERGE)
        for index in range(num_merged_videos):
            start_index = index * NUM_VIDEOS_TO_MERGE
            end_index = (index + 1) * NUM_VIDEOS_TO_MERGE
            if end_index > len(desired_options):
                if start_index > len(desired_options):
                    break
                else:
                    merger = desired_options[start_index:]
            else:
                merger = desired_options[start_index:end_index]
            merger_name = ''
            for opt_id in merger:
                merger_name += str(opt_id) + '_'
            if PERMUTE_SPECIFIC:
                merger = np.random.permutation(merger)
            mergers.append((merger_name[:-1], [(video_name_dict[str(opt_id)]) for opt_id in merger]))

    for merger in mergers:
        merge_videos_to_matrix(
            video_paths=merger[1],
            out_path=os.path.join(STORE_DIR, merger[0]),
            n_cols=N_COLS,
            auto_repeat=True
        )




