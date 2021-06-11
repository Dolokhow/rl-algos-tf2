import os
import logging
import time
import json

import tensorflow as tf
import numpy as np

from rltf2.agents.agent import Agent
from rltf2.utils.env import EnvInterface, RenderDummy, Renderer
from rltf2.utils.replay_buffer import get_replay_buffer, ReplayBufferProto
from rltf2.utils.file_io import yaml_to_dict, dict_to_yaml, check_create_dir, \
    create_dir, tensorboard_structured_summaries, get_logger, configure_tf_checkpoints


class Trainer:
    def __init__(self, env: EnvInterface, agent, train_param_yml_path, root_dir, renderer=None, eval_env=None, name=None):
        self.trial_name = name if name is not None else agent.tb_panel
        self.TB_GENERAL_PANEL = 'general'
        self.TB_TRAIN_PANEL = self.TB_GENERAL_PANEL + '/train'
        self.TB_EVAL_PANEL = self.TB_GENERAL_PANEL + '/eval'
        self.log_var = 'INIT'

        self.log_dir, self.weights_dir = self._init_store_dirs(root_dir=root_dir, trial_name=self.trial_name)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.logger = get_logger(name=self.trial_name, log_dir=self.log_dir)

        self.train_params_path = train_param_yml_path

        self.original_params = self._load_params()
        self.train_params = self._parse_train_args()
        self.evaluation_params = self._parse_eval_args()
        self.evaluate = bool(self.evaluation_params)
        if self.evaluate is False and agent.full_ep_options is True:
            self.logger.warning('{0: <5} :: No evaluation parameters specified but agent uses full episode options '
                                'that would benefit from evaluation.'.format(self.log_var))
        self._store_args()

        self.batch_size = self.train_params['batch_size']
        if self.train_params['data']['train_method'] == 'TD':
            store_trajectories = False
            self.replay_buffer: ReplayBufferProto = get_replay_buffer(
                rb_type=self.train_params['data']['type'],
                storage_size=self.train_params['data']['capacity'],
                params_dict={
                    'obs_dtype': agent.input_dtype,
                    'act_dtype': agent.action_dtype,
                    'default_dtype': agent.input_dtype,
                    'env': {"obs": {"shape": agent.obs_shape},
                            "act": {"shape": env.get_action_shape()},
                            "rew": {},
                            "next_obs": {"shape": agent.obs_shape},
                            "done": {}}
                }
            )
        else:
            self.replay_buffer = None
            store_trajectories = True

        self.train_runner: Runner = Runner(
            env=env,
            agent=agent,
            total_steps=self.train_params['total_steps'],
            intervals={
                'update': self.train_params['step_update_interval'],
                'store': self.train_params['step_store_interval'],
                'summary': self.train_params['step_summary_interval']
            },
            tracker=Tracker(
                average_factor=self.train_params['average_factor'],
                option_ids=agent.get_all_option_ids(),
                store_trajectories=store_trajectories,
                tb_panel=self.TB_TRAIN_PANEL
            ),
            renderer=renderer if renderer is not None else RenderDummy(),
            writer=self.summary_writer,
            opt_persist=False,
            max_steps=self.train_params['max_steps_per_ep'],
            warmup_steps=self.train_params['warmup_steps'],
            replay_buffer=self.replay_buffer,
            ckpt_path=self.weights_dir,
            tb_panel=self.TB_TRAIN_PANEL,
            logger=self.logger
        )
        self.agent: Agent = agent

        if self.evaluate is True:
            self.eval_runner: Runner = Runner(
                env=eval_env if eval_env is not None else env.get_copy(),
                agent=agent,
                total_steps=self.evaluation_params['total_steps'],
                # Lack of update interval indicates evaluation run! Lack of summaries interval indicates that
                # no summaries should be recorded (this is deferred to the Trainer class)
                intervals={},
                tracker=Tracker(
                    average_factor=self.train_params['average_factor'],
                    option_ids=agent.get_all_option_ids(),
                    store_trajectories=False,
                    tb_panel=self.TB_EVAL_PANEL
                ),
                renderer=renderer if renderer is not None else RenderDummy(),
                writer=None,
                opt_persist=self.agent.full_ep_options,
                max_steps=self.evaluation_params['max_steps_per_ep'],
                tb_panel=self.TB_EVAL_PANEL,
                logger=self.logger
            )
        else:
            self.eval_runner = None

    def _init_store_dirs(self, root_dir, trial_name):
        status = check_create_dir(dir_path=root_dir)
        if status is False:
            raise ValueError('{0: <5} :: Root directory path root_directory specified is not a valid path.'
                             .format(self.log_var))
        root_dir = os.path.join(root_dir, trial_name.replace(' ', '_'))
        status, root_dir = create_dir(path=root_dir, unique_tag=True)
        if status is False:
            raise ValueError('{0: <5} :: Could not configure root storing directory '
                             'with specified store_directory and trial_name.'.format(self.log_var))

        log_dir = os.path.join(root_dir, 'logs')
        os.mkdir(log_dir)

        weights_dir = os.path.join(root_dir, 'weights')
        os.mkdir(weights_dir)
        return log_dir, weights_dir

    def _load_params(self):
        params = yaml_to_dict(file_path=self.train_params_path)
        if not isinstance(params, dict):
            raise ValueError('{0: <5} :: Reading contents from configuration file failed.'.format(self.log_var))
        return params

    def _parse_single_arg(self, params_dict, arg, required=False, arg_type=None):
        if arg in params_dict:
            arg_val = params_dict[arg]
            if arg_type is None:
                return arg_val
            elif arg_type == int:
                return int(arg_val)
            elif arg_type == str:
                return str(arg_val)
            elif arg_type == float:
                return float(arg_val)
        else:
            if required is True:
                raise ValueError('{0: <5} :: Required parameter {1: <16} not correctly specified in yaml in '
                                 'context of other specified parameters.'.format(self.log_var, arg))
            else:
                self.logger.info('{0: <5} :: Optional parameter {1: <16} not specified in yaml. '
                                 'Calculating or assuming default value.'.format(self.log_var, arg))
                return None

    def _parse_train_args(self):
        train_sect = self._parse_single_arg(params_dict=self.original_params, arg='train', required=True, arg_type=None)
        # Trainer, and all other classes that handle env agent interactions function in terms of
        # steps by design. If total_steps is not provided, then total_episodes and max_steps_per_ep must be!
        total_episodes = self._parse_single_arg(params_dict=train_sect, arg='total_episodes', required=False, arg_type=int)
        if total_episodes is not None:
            max_steps_required = True
        else:
            max_steps_required = False
        max_steps_per_ep = self._parse_single_arg(
            params_dict=train_sect,
            arg='max_steps_per_ep',
            required=max_steps_required,
            arg_type=int
        )
        if max_steps_per_ep is not None and total_episodes is not None:
            total_steps = max_steps_per_ep * total_episodes
        else:
            total_steps = self._parse_single_arg(params_dict=train_sect, arg='total_steps', required=True, arg_type=int)
        if max_steps_per_ep is None:
            max_steps_per_ep = total_steps

        batch_size = self._parse_single_arg(params_dict=train_sect, arg='batch_size', required=False, arg_type=int)
        if batch_size is None:
            batch_size = 256

        warmup_steps_candidate = self._parse_single_arg(params_dict=train_sect, arg='warmup_steps', required=False, arg_type=int)
        if warmup_steps_candidate is None:
            warmup_steps_candidate = batch_size
        warmup_steps = max(warmup_steps_candidate, batch_size)

        step_update_interval = self._parse_single_arg(params_dict=train_sect, arg='step_update_interval', required=False, arg_type=int)
        if step_update_interval is None:
            step_update_interval = 1

        step_store_interval = self._parse_single_arg(params_dict=train_sect, arg='step_store_interval', required=False, arg_type=int)
        if step_store_interval is None:
            step_store_interval = int(total_steps / 10)

        step_summary_interval = self._parse_single_arg(params_dict=train_sect, arg='step_summary_interval', required=False, arg_type=int)
        if step_summary_interval is None:
            step_summary_interval = int(total_steps / 100)

        average_factor = self._parse_single_arg(params_dict=train_sect, arg='average_factor', required=False, arg_type=int)
        if average_factor is None:
            average_factor = 10

        ret_data_dict = {}
        data_sect = self._parse_single_arg(params_dict=train_sect, arg='data', required=True, arg_type=None)
        train_method = self._parse_single_arg(params_dict=data_sect, arg='train_method', required=True, arg_type=str)
        ret_data_dict['train_method'] = train_method
        capacity = self._parse_single_arg(params_dict=data_sect, arg='capacity', required=True, arg_type=int)
        ret_data_dict['capacity'] = capacity
        # Replay buffer is required which differentiates implementation types
        if train_method == 'TD':
            storage_type = self._parse_single_arg(params_dict=data_sect, arg='type', required=False, arg_type=str)
            if storage_type is None:
                storage_type = 'cpprb'
            ret_data_dict['type'] = storage_type

        self.logger.info('{0: <5} :: Train parameters successfully parsed.'.format(self.log_var))
        ret_train_dict = {
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
            'max_steps_per_ep': max_steps_per_ep,
            'batch_size': batch_size,
            'step_update_interval': step_update_interval,
            'step_store_interval': step_store_interval,
            'step_summary_interval': step_summary_interval,
            'average_factor': average_factor,
            'data': ret_data_dict
        }
        return ret_train_dict

    def _parse_eval_args(self):
        eval_sect = self._parse_single_arg(params_dict=self.original_params, arg='evaluation', required=False, arg_type=None)
        if eval_sect is not None:
            total_episodes = self._parse_single_arg(params_dict=eval_sect, arg='total_episodes', required=False, arg_type=int)
            if total_episodes is not None:
                max_steps_required = True
            else:
                max_steps_required = False
            max_steps_per_ep = self._parse_single_arg(
                params_dict=eval_sect,
                arg='max_steps_per_ep',
                required=max_steps_required,
                arg_type=int
            )
            if max_steps_per_ep is not None and total_episodes is not None:
                total_steps = max_steps_per_ep * total_episodes
            else:
                total_steps = self._parse_single_arg(params_dict=eval_sect, arg='total_steps', required=True, arg_type=int)
            if max_steps_per_ep is None:
                max_steps_per_ep = total_steps

            num_top_options = self._parse_single_arg(params_dict=eval_sect, arg='num_top_options', required=False, arg_type=int)
            if num_top_options is None:
                num_top_options = 1

            update_eval_interval = self._parse_single_arg(params_dict=eval_sect, arg='update_eval_interval', required=True, arg_type=int)
            average_factor = self._parse_single_arg(params_dict=eval_sect, arg='average_factor', required=False, arg_type=int)
            if average_factor is None:
                average_factor = 10
            self.logger.info('{0: <5} :: Evaluation parameters successfully parsed.'.format(self.log_var))
            ret_eval_dict = {
                'total_steps': total_steps,
                'max_steps_per_ep': max_steps_per_ep,
                'num_top_options': num_top_options,
                'update_eval_interval': update_eval_interval,
                'average_factor': average_factor
            }
        else:
            ret_eval_dict = {}

        return ret_eval_dict

    def _store_args(self):
        path = os.path.join(self.weights_dir, self.trial_name + '.yml')
        args_dict = {
            'train': self.train_params,
            'evaluation': self.evaluation_params
        }
        dict_to_yaml(data=args_dict, file_path=path)

    def _store_summaries(self, summaries):
        tensorboard_structured_summaries(
            writer=self.summary_writer,
            summaries=summaries,
            step=self.train_runner.total_step
        )

    def update(self):
        if self.replay_buffer is not None:
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            loss_summaries = self.agent.update(
                batch_obs=batch[0],
                batch_act=batch[1],
                batch_next_obs=batch[2],
                batch_rew=batch[3],
                batch_done=batch[4]
            )
        else:
            # TODO: Implement support for Policy Optimization approach where agent is updated based
            #  off of trajectory and reward histories. Use Tracker class for that!
            loss_summaries = None

        for summary in loss_summaries:
            summary[0] = self.TB_GENERAL_PANEL + '/' + summary[0]
        return loss_summaries

    def train(self):
        num_updates = 0
        train_done = False
        while not train_done:
            self.log_var = 'TRAIN'
            train_done = self.train_runner.run()

            if not train_done:
                loss_summaries = self.update()
                num_updates += 1
                train_tracker_history = os.path.join(self.log_dir, 'train_tracker.json')

                if self.train_runner.total_step % self.train_params['step_summary_interval'] == 0:
                    self._store_summaries(summaries=loss_summaries)
                    with open(train_tracker_history, 'w') as train_history:
                        json.dump(
                            {
                                'returns_history': self.train_runner.tracker.returns_history,
                                'options_history': self.train_runner.tracker.options_history
                            },
                            train_history,
                            indent=4
                        )
                    self.logger.info('{0: <5} :: STEP: {1: >7}/{2: <7} :: Storing summaries.'.format(
                        self.log_var, self.train_runner.total_step, self.train_runner.total_steps))
                if self.evaluate is True and num_updates % self.evaluation_params['update_eval_interval'] == 0:
                    self.log_var = 'EVAL'
                    if self.agent.full_ep_options is True:
                        options_to_test = self.train_runner.tracker.options_ranking[:self.evaluation_params['num_top_options']]
                    else:
                        options_to_test = [self.agent.get_cur_option_id()]

                    for option_id in options_to_test:
                        self.eval_runner.cur_opt_id = option_id
                        _ = self.eval_runner.run()
                        # Could be a hard reset as well, this way it will keep track of returns and options history
                        # through all evaluation calls, ie. it's Tracker will not be reset!
                        # self.eval_runner.soft_reset()
                    eval_summaries = self.eval_runner.tracker.get_summaries()
                    self._store_summaries(summaries=eval_summaries)


# Handles running the episodes. Handles agent - environment interactions.
# Uses Tracker class to keep track of agent performance metrics.
class Runner:
    def __init__(self, env, agent, total_steps, intervals, tracker=None, renderer=None, writer=None, max_steps=None,
                 warmup_steps=0, opt_persist=False, replay_buffer=None, ckpt_path=None, summaries_path=None,
                 tb_panel=None, logger=None):
        self.log_var = 'INIT'
        self.logger = logger if logger is not None else logging.getLogger(name=tb_panel)
        self.tb_panel = tb_panel if tb_panel is not None else agent.name

        self.opt_persist = opt_persist
        self.total_steps = total_steps
        self.max_steps = max_steps if max_steps is not None else total_steps
        self.warmup_steps = warmup_steps

        self.agent: Agent = agent
        self.env: EnvInterface = env
        self.obs = self._init_opt_obs()
        # Can be None even if Runner object is being used in Train capacity.
        # Needs to store trajectories in Tracker then!
        self.replay_buffer: ReplayBufferProto = replay_buffer
        self.tracker: Tracker = tracker if tracker is not None else Tracker(
            average_factor=5,
            option_ids=self.agent.get_all_option_ids(),
            store_trajectories=False if replay_buffer is not None else True,
            tb_panel=self.tb_panel
        )
        self.renderer: Renderer = renderer if renderer is not None else RenderDummy()

        if ckpt_path is not None:
            self.ckpt, self.ckpt_manager = configure_tf_checkpoints(policy=self.agent, dir_path=ckpt_path)
        else:
            self.ckpt = None
            self.ckpt_manager = None

        if 'update' in intervals:
            self.update_interval = intervals['update']
            self.test = False
        else:
            self.update_interval = None
            self.test = True

        if 'summary' in intervals:
            self.summary_interval = intervals['summary']
            self.defer_summaries = False
            if writer is None and summaries_path is None:
                raise ValueError('{0: <5} :: Neither summaries path nor summary writer provided, but summaries required'
                                 '. Please specify at least one.'.format(self.log_var))
            else:
                self.summary_writer: tf.summary.SummaryWriter = writer if writer is not None \
                    else tf.summary.create_file_writer(summaries_path)
        else:
            self.summary_interval = None
            self.summary_writer = None
            self.defer_summaries = True

        if 'store' in intervals:
            self.store_interval = intervals['store']
            self.defer_storing = False
            if ckpt_path is not None:
                self.ckpt, self.ckpt_manager = configure_tf_checkpoints(policy=self.agent, dir_path=ckpt_path)
            else:
                raise ValueError('{0: <5} :: No checkpoint path provided but intervals dict suggests checkpoints should'
                                 ' be saved. Please specify ckpt_path.'.format(self.log_var))
        else:
            self.store_interval = None
            self.defer_storing = True

        if self.test is True:
            self.log_var = 'TEST'
        else:
            self.log_var = 'TRAIN'

        # Counters

        self.cur_opt_id = self.agent.get_cur_option_id()
        self.total_step = 0
        self.episode_step = 0
        self.total_episode = 0
        self.ep_start_t = time.perf_counter()

    def load_agent(self, ckpt_path):
        checkpoint = tf.train.Checkpoint(self.agent)
        # Restore the checkpointed values to the `agent` object.
        checkpoint.restore(ckpt_path)

    # TODO: See how to handle this.
    def load_history(self, history_path):
        pass

    def _init_opt_obs(self):
        obs = self.env.env_reset()
        mod_obs, _ = self.agent.modify_observation(obs=obs)
        return mod_obs

    def select_action(self, obs, test):
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        return self.agent.select_action(obs=tf.constant(obs), test=test)

    def _take_step(self, action):
        obs, reward, done, markov_done, env_args = self.env.env_action(action=action, episode_step=self.episode_step)
        obs, option_id = self.agent.modify_observation(obs=obs)
        return obs, reward, option_id, done, markov_done, env_args

    def _update_counters(self, done):
        self.total_step += 1
        self.episode_step += 1
        episode_step = self.episode_step

        if bool(done) is True or self.episode_step >= self.max_steps:
            done = True
            elapsed_time = time.perf_counter() - self.ep_start_t
            fps = (self.episode_step - 1) / elapsed_time
            elapsed_time *= 1000
            self.ep_start_t = time.perf_counter()

            self.episode_step = 0
            self.total_episode += 1
        else:
            fps = None
            elapsed_time = None

        return done, episode_step, fps, elapsed_time

    def soft_reset(self):
        self.total_step = 0
        self.episode_step = 0
        self.total_episode = 0
        self.ep_start_t = time.perf_counter()
        self.agent.on_new_episode()
        self.obs = self._init_opt_obs()
        self.cur_opt_id = self.agent.get_cur_option_id()

    def hard_reset(self):
        self.soft_reset()
        self.tracker.reset()

    def _log_progress(self, origin, info_dict):
        if self.logger is not None:
            # self.logger.info('{0: <5} :: {1: >4}/{2: <4} episodes FULLY finished. '
            #                  'MEAN REWARD: {3: >10.2f}.'.format(self.LOG_ORIGIN, len(self.ep_returns_ev),
            #                                                     self.max_ep_ev, float(mean_episode_reward.numpy())))
            if origin == 'episode_end':
                if info_dict['option_id'] != '':
                    option_segment = 'Opt ID: ' + info_dict['option_id'] + '.'
                else:
                    option_segment = info_dict['option_id']
                self.logger.info('{0: <5} :: STEP: {1: >7}/{2: <7} :: Episode {3: <6} finished @ {4: <6} step. '
                                 'REWARD: {5: >10.2f}. {6: <12} FPS: {7: >8.2f}. ABS TIME [ms]: {8: >10.1f}. '
                                 'TERMINAL STATE: {9}.'.format(
                                    self.log_var, self.total_step, self.total_steps, self.total_episode,
                                    info_dict['ep_step'], info_dict['returns'], option_segment, info_dict['fps'],
                                    info_dict['elapsed_time'], info_dict['true_done']))
            elif origin == 'store_interval':
                self.logger.warning('{0: <5} :: Store interval not provided but ckpt_path suggests that '
                                    'storing should be done. Assuming store_interval to be the same as summary_interval'
                                    '.'.format(self.log_var))

    def _store_summaries(self):
        summaries = self.tracker.get_summaries()
        tensorboard_structured_summaries(
            writer=self.summary_writer,
            summaries=summaries,
            step=self.total_step
        )

    def run(self):
        # If the agent was being handled by some other runner object in the meantime
        self.agent.set_cur_option_id(option_id=self.cur_opt_id)
        while self.total_step < self.total_steps:
            if self.defer_summaries is False and self.total_step % self.summary_interval == 0:
                self._store_summaries()
            if self.defer_storing is False and self.total_step % self.store_interval == 0:
                self.ckpt_manager.save()

            self.renderer.render_frame(env=self.env.env, option_id=self.cur_opt_id, total_reward=self.tracker.returns_ep)
            action = self.select_action(obs=self.obs, test=self.test)
            next_obs, reward, option_id, done, markov_done, _ = self._take_step(action=action)
            # If one wants to bootstrap (overwrite environment max steps param if environment has one)
            # we should pass markov_done instead of done. The rest of the code should work as is,
            # except in a special case where terminal condition overlaps
            # with the non markov end criterion (env max steps). This is the reason done and markov_done
            # are kept as separate variables.
            ep_finished, ep_step, fps, elapsed_time = self._update_counters(done=done)

            returns = self.tracker.update_metrics(
                reward=reward,
                option_id=option_id,
                episode_id=self.total_episode,
                total_step=self.total_step,
                done=ep_finished,
                is_natural=ep_finished == bool(markov_done)
            )
            if self.replay_buffer is not None:
                self.replay_buffer.add(obs=self.obs, action=action, next_obs=next_obs, reward=reward, done=markov_done)

            if ep_finished:
                self.agent.on_new_episode()
                if self.opt_persist is True:
                    self.agent.set_cur_option_id(option_id=option_id)
                self.obs = self._init_opt_obs()

                self._log_progress(
                    origin='episode_end',
                    info_dict={
                        'ep_step': ep_step,
                        'fps': fps,
                        'elapsed_time': elapsed_time,
                        'returns': returns,
                        'option_id': str(option_id) if self.agent.full_ep_options is True and self.agent.num_options > 1
                        else '',
                        'true_done': 'True' if bool(markov_done) is True else 'False'
                    }
                )
            else:
                self.obs = next_obs

            # If it's time for agent update we relinquish control to the caller to
            # handle the update.
            if self.test is False and self.total_step > self.warmup_steps and self.total_step % self.update_interval == 0:
                # Saving this option id for when this method is being called again
                self.cur_opt_id = self.agent.get_cur_option_id()
                return False

        if self.total_step >= self.total_steps:
            self.soft_reset()
            return True


# TODO: Implement optional buffer that keeps track of past trajectories and rewards!
class Tracker:
    def __init__(self, average_factor=1, option_ids=None, store_trajectories=False, tb_panel=None):

        self.tb_panel = tb_panel
        self.store_trajectories = store_trajectories
        if option_ids is None:
            self.option_ids = [0]
        else:
            self.option_ids = option_ids
        self.average_factor = average_factor

        self.returns_ep = []
        self.options_ep = []

        self.returns_history = []
        self.options_history = {}
        self.options_ranking = []

        self.options_common_count = 0
        self.returns_count = 0

        for opt_id in self.option_ids:
            self.options_history[opt_id] = {
                # Episode ids
                'episodes': [],
                # Cumulative end step of episode id
                'total_step': [],
                # Number of steps in episode
                'num_steps': [],
                # Percentage of steps in episode
                'rel_steps': [],
                # Total reward during the option
                'returns': [],
                # Projected reward per option
                'proj_returns': []
            }

    def reset(self):
        self.returns_ep = []
        self.options_ep = []

        self.returns_history = []
        self.options_history = {}
        self.options_ranking = []

        self.options_common_count = 0
        self.returns_count = 0

        for opt_id in self.option_ids:
            self.options_history[opt_id] = {
                # Episode ids
                'episodes': [],
                # Cumulative end step of episode id
                'total_step': [],
                # Number of steps in episode
                'num_steps': [],
                # Percentage of steps in episode
                'rel_steps': [],
                # Total reward during the option
                'returns': [],
                # Projected reward per option
                'proj_returns': []
            }

    def update_metrics(self, reward, option_id, episode_id, total_step, done, is_natural):
        self.returns_ep.append(reward)
        self.options_ep.append(option_id)
        cumm_ep_return = np.sum(self.returns_ep)

        if done is True:
            # Update reward history
            self.returns_history.append(cumm_ep_return)
            self.returns_count += 1

            # Update option history & refresh option metrics
            option_steps = np.array(self.options_ep)
            reward_steps = np.array(self.returns_ep)
            options_ranking = []
            options_counts_next_iter = []

            for key, _ in self.options_history.items():
                key_indices = np.where(option_steps == key)[0]
                key_steps = key_indices.shape[0]
                # if option is not used during the episode, we are not storing it!
                if key_steps != 0:
                    key_rewards = reward_steps.take(key_indices)
                    total_key_reward = np.sum(key_rewards)
                    proj_key_reward = (total_key_reward / key_steps) * reward_steps.shape[0]

                    self.options_history[key]['num_steps'].append(key_steps)
                    self.options_history[key]['rel_steps'].append(key_steps / reward_steps.shape[0])
                    self.options_history[key]['returns'].append(total_key_reward)
                    self.options_history[key]['episodes'].append(episode_id)
                    self.options_history[key]['total_step'].append(total_step)
                    self.options_history[key]['proj_returns'].append(proj_key_reward)

                # Updated current option usage
                options_counts_next_iter.append(len(self.options_history[key]['returns']))

                if len(self.options_history[key]['returns']) > 0 and self.options_common_count > 0:
                    average_factor = min(self.average_factor, self.options_common_count)
                    performance = self.options_history[key]['returns'][-average_factor:]
                    average_performance = np.mean(performance)
                    options_ranking.append((key, average_performance))

            counts = np.array(options_counts_next_iter)
            self.options_common_count = np.min(counts[np.nonzero(counts)])

            options_ranking = sorted(options_ranking, key=lambda entry: entry[1], reverse=True)
            self.options_ranking = [entry[0] for entry in options_ranking]

            # Reset counters
            if self.store_trajectories:
                # TODO: Implement trajectory record tracking here. Uses requires is_natural.
                pass
            self.returns_ep = []
            self.options_ep = []

        return cumm_ep_return

    def update_complete_history(self, obs, next_obs, reward, option_id, total_step, episode_id, done, is_natural):
        if self.store_trajectories is True:
            # TODO: Implement trajectory tracking.
            pass
        self.update_metrics(
            reward=reward,
            option_id=option_id,
            episode_id=episode_id,
            total_step=total_step,
            done=done,
            is_natural=is_natural
            )

    def get_summaries(self):
        summaries = []
        average_factor = min(self.returns_count, self.average_factor)
        if average_factor > 0:
            returns = self.returns_history[-average_factor:]
            mean_returns = np.mean(returns)
            summary_id = 'ep_return[' + str(int(self.average_factor)) + ']'
            if self.tb_panel is not None:
                summary_id = self.tb_panel + '/' + summary_id
            summary = [summary_id, 'scalar', mean_returns]
            summaries.append(summary)

            # for each option, and for each metric we are tracking across options we average past
            # num_episode_returns results, but ONLY FOR EPISODES WHERE THE OPTION WAS ACTUALLY USED!
            # checking len(lst_data) checks whether the option was used at all thus far!
            for option_id, option_metrics in self.options_history.items():
                for metric_id, lst_data in option_metrics.items():
                    if len(lst_data) > 0:
                        average_factor = min(self.options_common_count, self.average_factor)
                        metric = lst_data[-average_factor:]
                        mean_metric = np.mean(metric)
                        if self.tb_panel is not None:
                            split = self.tb_panel.split('/')
                            last_id = split[-1]
                        else:
                            last_id = ''
                        summary_id = 'opt_decomp_' + metric_id + '/id_' + str(option_id) + '/' + metric_id + '[' + \
                                     str(int(self.average_factor)) + ']' + '/' + last_id
                        # summary_id = 'options/id_' + str(option_id) + '/' + metric_id + '[' + \
                        #              str(int(self.average_factor)) + ']'
                        summary = [summary_id, 'scalar', mean_metric]
                        summaries.append(summary)
        return summaries

