import tensorflow as tf
import numpy as np
import os
import gym
from abc import abstractmethod
import time

from rltf2.utils.replay_buffer import get_replay_buffer, ReplayBufferProto
from rltf2.agents.agent import Agent
from rltf2.utils.file_io import copy_file, yaml_to_dict, check_create_dir, \
    create_dir, split_path, tensorboard_structured_summaries, configure_tf_checkpoints, get_logger


class Experiment:
    # Folder names
    LOG_DIR = 'logs'
    WEIGHTS_DIR = 'weights'
    LOG_ORIGIN = 'INIT'
    # Dictionary list names
    EP_LST_NAME = 'episodes'
    REW_LST_NAME = 'returns'
    STEP_LST_NAME = 'num_steps'
    REL_STEP_LST_NAME = 'percent_steps'
    REW_PROJ_LST_NAME = 'projected_returns'
    # Tensorboard panels
    GENERAL_PANEL = 'general_diagnostics/'
    OPTION_PANEL = 'option_diagnostics/'

    def __init__(self, env, agent, config_path, store_dir, eval_env=None, name=None, render_mode='default'):

        # self.__LOG_DIR = 'logs'
        # self.__WEIGHTS_DIR = 'weights'
        # self.LOG_ORIGIN = 'INIT'

        self.env = env
        # Render mode can be either 'default' for default rendering mechanism of environment eg OpenAI Gym
        # or 'custom' to be used in custom render technique implementation.
        if render_mode is None:
            self.render = False
        else:
            self.render = True
            assert render_mode in ('default', 'custom')
            self.render_mode = render_mode
        self.agent: Agent = agent

        self.agent_input_dtype = self.agent.input_dtype
        self.agent_value_dtype = self.agent.input_dtype
        self.agent_action_dtype = self.agent.action_dtype

        self.num_options = self.agent.num_options
        self.all_option_ids = self.agent.get_all_option_ids()
        self.cur_option = self.agent.get_cur_option_id()
        # Option performance measure. Will be modified only for agents that both use options and these options span
        # across entire episodes. Otherwise, if agent does not use options or switches options within an episode, this
        # attribute will not change value from its initial one. Distinction between the two cases is regulated by
        # self.separate_eval_per_option.
        self.cur_opt_performance_comparison = [self.cur_option]

        self.config_path = config_path
        params = self._init_params()
        if name is not None:
            self.trial_name = name
        else:
            self.trial_name = self.agent.name
        self.store_dir = store_dir
        self.log_dir, self.weights_dir = self._init_store_dirs()
        self.logger = get_logger(name=self.trial_name, log_dir=self.log_dir)

        # ############ TRAIN ARGS ############
        self.max_step_tr, self.max_ep_tr, self.warmup_steps, self.max_steps_per_ep, \
            self.agnt_updt_interval, self.batch_size = self._parse_train_args(params=params)

        # Episode start time. To be set and reset within a train method. Does restart with new episodes.
        self.ep_start_t_tr = None
        # Total step in a training procedure, does not reset with new episodes
        self.cum_step_tr = 0
        # Total episodes in training procedure, does not reset with new episodes
        self.cum_ep_tr = 0

        # Current episode step, does reset
        self.cur_ep_step_tr = 0
        # Current episode return, does reset
        self.cur_ep_return_tr = 0

        # Current episode decomposed returns, does reset
        self.cur_ep_decomp_return_tr = []
        self.cur_ep_decomp_opts_tr = []

        # All returns during training, does not reset with new episodes
        self.ep_returns_tr = []
        # All full-episode options used during training, does not reset with new episodes
        self.ep_options_tr = {}
        for opt_id in self.all_option_ids:
            self.ep_options_tr[opt_id] = {
                # Episode ids
                self.EP_LST_NAME: [],
                # Number of steps in episode
                self.STEP_LST_NAME: [],
                # Percentage of steps in episode
                self.REL_STEP_LST_NAME: [],
                # Total reward during the option
                self.REW_LST_NAME: [],
                # Projected reward per option
                self.REW_PROJ_LST_NAME: []
            }

        # Train initial environment obs. If environment has random initial state, it is going to be handled
        # within the _env_reset() function
        self.init_obs_tr = self._wrapped_env_reset(test=False)

        # ############ EVALUATION ARGS #######
        self.separate_eval_per_option = True if self.agent.full_ep_options and self.num_options > 1 else False
        # Whether to do a separate evaluation
        self.eval_agnt, self.max_ep_ev, self.max_step_ev, self.max_opt_ev, self.stricter_exit_ev, self.agnt_ev_interval\
            = self._parse_eval_args(params=params)

        # Episode start time. To be set and reset within an evaluation method. Does restart with new episodes.
        self.ep_start_t_ev = None
        # Total step in a eval procedure, does not reset with new episodes
        self.cum_step_ev = 0
        # Current evaluation episode step, does reset
        self.cur_ep_step_ev = 0
        # Current evaluation episode, does reset
        self.cur_ep_ev = 0
        # Current evaluation episode return, does reset
        self.cur_ep_return_ev = 0
        # All returns during single evaluation call, does reset with new evaluation calls
        self.ep_returns_ev = []

        # If no separate evaluation is required, to be left as None
        if eval_env is None and self.eval_agnt is True:
            self.eval_env = self._copy_env()
            # Eval initial environment obs. If environment has random initial state, it is going to be handled
            # within the _env_reset() function.
            self.init_obs_ev = self._wrapped_env_reset(test=True)
        else:
            self.eval_env = eval_env

        # ############ REPLAY BUFFER ARGS ####
        # Replay buffer size
        self.rb_cap, self.rb_type = self._parse_rb_args(params=params)
        # Must be here since we need to add experience to it from the train loop.
        # For now it is the same ReplayBuffer object that is assigned to each agent
        self.rb: ReplayBufferProto = get_replay_buffer(
            rb_type=self.rb_type,
            storage_size=self.rb_cap,
            params_dict={
                'obs_dtype': self.agent_input_dtype,
                'act_dtype': self.agent_action_dtype,
                'default_dtype': self.agent_input_dtype,
                'env': {"obs": {"shape": self.agent.obs_shape},
                        "act": {"shape": self.env.action_space.shape},
                        "rew": {},
                        "next_obs": {"shape": self.agent.obs_shape},
                        "done": {}}
            }
        )

        # ############ STORING & VISUALIZATION ARGS ##########
        self.summary_interval, self.store_interval, \
            self.num_episode_returns, self.store_graph = self._parse_logging_args(params=params)

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.checkpoint, self.checkpoint_manager = self._init_checkpoints()
        self.logger.info('{0: <5} :: SUCCESSFULLY CREATED EXPERIMENT: ID: {1}.'.format(
                         self.LOG_ORIGIN, self.trial_name))

    def _parse_train_args(self, params):
        # Stricter of the two arguments below will be used
        # Number of episodes to train the agent for
        max_ep_tr = params['train']['max_episodes']
        # Number of steps to train the agent for
        max_step_tr = params['train']['max_steps']
        if max_ep_tr is None and max_step_tr is None:
            raise ValueError('INITIALIZATION: Both training max_episodes and max_steps parameters are None. '
                             'At least one should be specified.')
        else:
            if max_ep_tr is None:
                max_step_tr = int(max_step_tr)
                max_ep_tr = max_step_tr
                self.logger.info('{0: <5} :: Training max_episodes not specified. '
                                 'Assuming very large (infinite) value.'.format(self.LOG_ORIGIN))
            if max_step_tr is None:
                max_ep_tr = int(max_ep_tr)
                max_step_tr = int(1e9)
                self.logger.info('{0: <5} :: Training max_steps not specified. '
                                 'Assuming very large (infinite) value.'.format(self.LOG_ORIGIN))

        # Model training interval in steps
        if params['train']['batch_size'] is not None:
            batch_size = int(params['train']['batch_size'])
        else:
            raise ValueError('INITIALIZATION: batch_size must be specified.')

        # Number of warmup steps for training where random actions are taken
        if params['train']['warmup_steps'] is None:
            warmup_steps = batch_size
            self.logger.warning('{0: <5} :: No warmup_steps specified. '
                                'Minimal warmup steps corresponding to batch size assumed.'.format(self.LOG_ORIGIN))
        elif params['train']['warmup_steps'] < batch_size:
            warmup_steps = batch_size
            self.logger.warning('{0: <5} :: Number of warmup steps is lower than batch_size. This may cause issues '
                                'with the replay buffer. warmup_steps set to batch_size.'.format(self.LOG_ORIGIN))
        else:
            warmup_steps = int(params['train']['warmup_steps'])

        # Number of steps allowed within the episode
        if params['train']['max_steps_per_episode'] is not None:
            max_steps_per_ep = int(params['train']['max_steps_per_episode'])
        else:
            if max_step_tr is not None:
                max_steps_per_ep = max_step_tr
                self.logger.warning('{0: <5} :: max_steps_per_episode not specified. '
                                    'Assuming it is the same as max_steps parameter. '
                                    'This may not be the desired behavior.'.format(self.LOG_ORIGIN))
            else:
                max_steps_per_ep = 3e6
                self.logger.warning('{0: <5} :: max_steps_per_episode not specified. '
                                    'Assuming a very large number 3e6. This may not be the desired '
                                    'behavior.'.format(self.LOG_ORIGIN))
        # Model training interval in steps
        if params['train']['agent_update_interval'] is not None:
            agnt_updt_interval = int(params['train']['agent_update_interval'])
        else:
            raise ValueError('INITIALIZATION: agent_update_interval must be specified.')

        return max_step_tr, max_ep_tr, warmup_steps, max_steps_per_ep, agnt_updt_interval, batch_size

    def _parse_eval_args(self, params):
        eval_agnt = True if 'evaluate' in params else False
        if eval_agnt is True:
            eval_agnt = params['evaluate']['separate_evaluation']
            if eval_agnt is False:
                self.logger.info('{0: <5} :: No separate evaluation will be performed.'.format(self.LOG_ORIGIN))
                max_ep_ev = None
                max_step_ev = None
                agnt_ev_interval = None
                stricter_exit_ev = None
                max_opt_ev = None
            else:
                # Stricter of the two arguments below will be used
                # Number of episodes to evaluate the agent for
                max_ep_ev = params['evaluate']['max_episodes']
                # Number of steps to evaluate the agent for
                max_step_ev = params['evaluate']['max_steps']
                if max_ep_ev is None and max_step_ev is None:
                    raise ValueError('INITIALIZATION: Both evaluation max_episodes and max_steps parameters are None. '
                                     'At least one should be specified.')
                else:
                    if max_ep_ev is None:
                        max_step_ev = int(max_step_ev)
                        max_ep_ev = max_step_ev
                        self.logger.info(
                            '{0: <5} :: Evaluation max_episodes not specified. Assuming very large (infinite) '
                            'value.'.format(self.LOG_ORIGIN))
                    if max_step_ev is None:
                        max_ep_ev = int(max_ep_ev)
                        max_step_ev = int(1e9)
                        self.logger.info(
                            '{0: <5} :: Evaluation max_steps not specified. Assuming very large (infinite) '
                            'value.'.format(self.LOG_ORIGIN))

                max_steps_acc_episode = max_ep_ev * self.max_steps_per_ep
                if max_steps_acc_episode < max_step_ev:
                    stricter_exit_ev = max_steps_acc_episode
                else:
                    stricter_exit_ev = max_step_ev

                # Model evaluation interval in steps
                if params['evaluate']['agent_eval_interval'] is not None:
                    agnt_ev_interval = int(params['evaluate']['agent_eval_interval'])
                else:
                    raise ValueError('INITIALIZATION: agent_eval_interval must be specified.')

                # For options using agents with full episode options
                if params['evaluate']['max_options'] is not None:
                    max_opt_ev = int(params['evaluate']['max_options'])
                elif self.separate_eval_per_option:
                    self.logger.warning(
                        '{0: <5} :: Evaluation max_options not specified, but agent with episode spanning options '
                        'is used. Setting max_options to 10% of total number of options.'.format(self.LOG_ORIGIN))
                    max_opt_ev = int(0.1 * self.num_options)
                else:
                    max_opt_ev = None

        else:
            max_ep_ev = None
            max_step_ev = None
            agnt_ev_interval = None
            stricter_exit_ev = None
            max_opt_ev = None
            self.logger.info(
                '{0: <5} :: No separate evaluation will be performed.'.format(self.LOG_ORIGIN))

        return eval_agnt, max_ep_ev, max_step_ev, max_opt_ev, stricter_exit_ev, agnt_ev_interval

    def _parse_rb_args(self, params):
        # Capacity of the replay buffer
        if params['replay_buffer']['capacity'] is None:
            raise ValueError('INITIALIZATION: Replay buffer capacity not specified.')
        else:
            rb_cap = params['replay_buffer']['capacity']
        if params['replay_buffer']['type'] is None:
            self.logger.info(
                '{0: <5} :: Replay buffer type not specified. Assuming CPPRB.'.format(self.LOG_ORIGIN))
            rb_type = 'cpprb'
        else:
            rb_type = params['replay_buffer']['type']
        return rb_cap, rb_type

    def _parse_logging_args(self, params):
        # Model summary interval
        if params['log_and_store']['summary_interval'] is not None:
            summary_interval = params['log_and_store']['summary_interval']
        else:
            raise ValueError('INITIALIZATION: Summary interval summary_interval must be specified.')

        # Model store interval measured in number of steps
        if params['log_and_store']['store_interval'] is not None:
            store_interval = params['log_and_store']['store_interval']
        else:
            store_interval = summary_interval
            self.logger.warning(
                '{0: <5} :: store_interval not specified. Assuming summary_interval.'.format(self.LOG_ORIGIN))

        # Number of last episodes to average returns for plotting
        if params['log_and_store']['num_episodes_mean_episode_return'] is not None:
            num_episode_returns = params['log_and_store']['num_episodes_mean_episode_return']
        else:
            num_episode_returns = 1

        # Whether to store agent graph visualization
        if params['log_and_store']['store_graph'] is not None:
            store_graph = params['log_and_store']['store_graph']
        else:
            store_graph = False

        return summary_interval, store_interval, num_episode_returns, store_graph

    def _init_params(self):
        params = yaml_to_dict(file_path=self.config_path)
        if not isinstance(params, dict):
            raise ValueError('Reading contents from configuration file failed.')
        return params

    def _init_store_dirs(self):
        status = check_create_dir(dir_path=self.store_dir)
        if status is False:
            raise ValueError('INITIALIZATION: Store directory path store_directory specified not a valid path.')
        root_dir = os.path.join(self.store_dir, self.trial_name.replace(' ', '_'))
        status, root_dir = create_dir(path=root_dir, unique_tag=True)
        if status is False:
            raise ValueError('INITIALIZATION: Could not configure root storing directory '
                             'with specified store_directory and trial_name.')
        _, config_file_name, _, _ = split_path(file_path=self.config_path)
        copy_file(src_path=self.config_path, dst_path=os.path.join(root_dir, config_file_name))

        log_dir = os.path.join(root_dir, self.LOG_DIR)
        os.mkdir(log_dir)

        weights_dir = os.path.join(root_dir, self.WEIGHTS_DIR)
        os.mkdir(weights_dir)

        return log_dir, weights_dir

    def _init_checkpoints(self):
        return configure_tf_checkpoints(
            policy=self.agent,
            dir_path=self.weights_dir
        )

    # def _store_summaries(self, summaries):
    #     tensorboard_structured_summaries(
    #         writer=self.summary_writer,
    #         summaries=summaries,
    #         step=self.cum_step_tr
    #     )

    def _store_summaries(self, test, loss_summaries=None):
        summaries = []
        if loss_summaries is not None:
            for summary in loss_summaries:
                summary[0] = self.GENERAL_PANEL + summary[0]
                summaries.append(summary)

        # _store_summaries is called from training procedure and we are storing training metrics
        if test is False:
            # num_episode_returns configures how many past episodes from the current one we are averaging when reporting
            if len(self.ep_returns_tr) > 0:
                if self.num_episode_returns < self.cum_ep_tr:
                    mean_ep_return = self.ep_returns_tr[-1]
                else:
                    mean_ep_return = self.ep_returns_tr[-self.num_episode_returns:]
                    mean_ep_return = np.mean(mean_ep_return)
                # mean_ep_return = tf.squeeze(mean_ep_return)
                summary_id = self.GENERAL_PANEL + 'mean_' + str(int(self.num_episode_returns)) + '_ep_return'
                summary = (summary_id, 'scalar', mean_ep_return)
                summaries.append(summary)

                # for each option, and for each metric we are tracking across options we average past
                # num_episode_returns results, but ONLY FOR EPISODES WHERE THE OPTION WAS ACTUALLY USED!
                # checking len(lst_data) checks whether the option was used at all thus far!
                for option_id, option_metrics in self.ep_options_tr.items():
                    for metric_id, lst_data in option_metrics.items():
                        if len(lst_data) > 0:
                            if self.num_episode_returns < len(lst_data):
                                mean_metric = lst_data[-1]
                            else:
                                mean_metric = lst_data[-self.num_episode_returns:]
                                mean_metric = np.mean(mean_metric)
                            summary_id = self.OPTION_PANEL + str(option_id) + '/mean_' + \
                                str(int(self.num_episode_returns)) + '_' + metric_id
                            summary = (summary_id, 'scalar', mean_metric)
                            summaries.append(summary)
        else:
            # TODO: Implement evaluation metric storing
            if len(self.ep_returns_ev) > 1 and self.stricter_exit_ev % self.max_steps_per_ep != 0:
                self.ep_returns_ev = self.ep_returns_ev[:-1]

            if len(self.ep_returns_ev) > 0:
                ep_reward = tf.convert_to_tensor(self.ep_returns_ev, dtype=tf.float32)
                if len(self.ep_returns_ev) > 1:
                    mean_episode_reward = tf.squeeze(tf.math.reduce_mean(ep_reward))
                else:
                    mean_episode_reward = tf.squeeze(ep_reward)
                self.logger.info('{0: <5} :: {1: >4}/{2: <4} episodes FULLY finished. '
                                 'MEAN REWARD: {3: >10.2f}.'.format(self.LOG_ORIGIN, len(self.ep_returns_ev),
                                                                    self.max_ep_ev, float(mean_episode_reward.numpy())))
                self._store_summaries(summaries=[('eval_mean_ep_return', 'scalar', mean_episode_reward)])

        tensorboard_structured_summaries(
            writer=self.summary_writer,
            summaries=summaries,
            step=self.cum_step_tr
        )

    def _store_model(self):
        self.checkpoint_manager.save()
        # TODO: Figure out if it is really necessary to serialize the model this way.
        #  Current issue is that tfp.distribution objects are not savable with tf.saved_model:
        #  TypeError: To be compatible with tf.eager.defun, Python functions must return zero or more Tensors;
        #  in compilation of <function GaussianLayer.call at 0x1576e0c80>, found return value of type
        #  <class 'tensorflow_probability.python.distributions.mvn_diag.MultivariateNormalDiag'>, which is not a Tensor.
        #  This will create the issue if we ever need to export the actual graph with frozen weights to be used in a
        #  third party app.
        # kwargs = self.agent.get_forward_pass_input_spec()
        # signatures = self.agent.forward_pass.get_concrete_function(**kwargs)
        # try:
        #     tf.saved_model.save(
        #         obj=self.agent,
        #         export_dir=self.weights_dir,
        #         signatures=signatures
        #     )
        # except TypeError as e:
        #     self.logger.warning('{0: <5} :: Failed to serialize model graph. Original ERROR MSG: {1}'
        #                         .format(self.LOG_ORIGIN, str(e)))

    def _take_step(self, action, test):
        obs, reward, done, env_args = self._env_action(action=action, test=test)
        obs, option_id = self.agent.modify_observation(obs=obs)
        if self.render is True:
            ret = self._env_render(test=test)
        # TODO: Move this to constructor and warn if selected max number of steps i greater than env.
        if (hasattr(self.env, "_max_episode_steps") and
                self.cur_ep_step_tr == self.env._max_episode_steps):
            done = 0.0
        return obs, reward, option_id, done, env_args

    def _select_action(self, obs, test):
        if self.cum_step_tr < self.warmup_steps:
            action = self._env_act_sample()
        else:
            action = self._env_act_select(obs=obs, test=test)
        return action

    def _calc_mean_ep_return(self):
        if self.num_episode_returns < self.cum_ep_tr:
            mean_ep_return = self.ep_returns_tr[-1]
        else:
            mean_ep_return = self.ep_returns_tr[-self.num_episode_returns:]
            mean_ep_return = tf.reduce_mean(mean_ep_return)
        mean_ep_return = tf.squeeze(mean_ep_return)
        return mean_ep_return

    def _update_current_counters(self):
        option_steps = np.array(self.cur_ep_decomp_opts_tr)
        reward_steps = np.array(self.cur_ep_decomp_return_tr)

        option_performance_ranking = []
        # Update option metrics
        for key, _ in self.ep_options_tr.items():
            key_indices = np.where(option_steps == key)
            key_steps = key_indices.shape[0]
            # if option is not used during the episode, we are not storing it!
            if key_steps != 0:
                key_rewards = reward_steps.take(key_indices)
                total_key_reward = np.sum(key_rewards)
                proj_key_reward = (total_key_reward / key_steps)*reward_steps.shape[0]

                self.ep_options_tr[key][self.STEP_LST_NAME].append(key_steps)
                self.ep_options_tr[key][self.REL_STEP_LST_NAME].append(key_steps/reward_steps.shape[0])
                self.ep_options_tr[key][self.REW_LST_NAME].append(total_key_reward)
                self.ep_options_tr[key][self.EP_LST_NAME].append(self.cum_ep_tr)
                self.ep_options_tr[key][self.REW_PROJ_LST_NAME].append(proj_key_reward)

            if self.separate_eval_per_option and len(self.ep_options_tr[key][self.REW_LST_NAME]) > 0:
                option_performance_ranking.append((key, self.ep_options_tr[key][self.REW_LST_NAME][-1]))

        # Store currently most performant options. For agents with full episode length options this will be used.
        # For agents without options, or agents that switch options withing episodes this will be disregarded.
        if self.separate_eval_per_option:
            option_performance_ranking = sorted(option_performance_ranking, key=lambda entry: entry[1], reverse=True)
            self.cur_opt_performance_comparison = [entry[0] for entry in option_performance_ranking]

        self.ep_returns_tr.append(np.sum(reward_steps))
        self.cur_ep_decomp_return_tr = []
        self.cur_ep_decomp_opts_tr = []
        self.cur_ep_step_tr = 0

        # TODO: Remove this line.
        self.cur_ep_return_tr = 0

    def _train_update(self, obs, reward, option_id, done):
        self.cum_step_tr += 1
        self.cur_ep_step_tr += 1
        if done is True or self.cur_ep_step_tr > self.max_steps_per_ep:
            # Signals option resampling for agents that use full-episode options
            self.agent.on_new_episode()

            end_step = self.cur_ep_step_tr
            elapsed_time = time.perf_counter() - self.ep_start_t_tr
            fps = (self.cur_ep_step_tr - 1) / elapsed_time
            elapsed_time *= 1000

            # self.ep_returns_tr.append(self.cur_ep_return_tr)
            # self.cur_ep_return_tr = 0
            self._update_current_counters()

            next_obs = self._wrapped_env_reset(test=False)
            if self.render is True:
                ret = self._env_render(test=False)

            self.cum_ep_tr += 1
            self.ep_start_t_tr = time.perf_counter()

            self.logger.info('{0: <5} :: Episode {1: >7}/{2: <7} finished @ {3: <5} step. REWARD: {4: >10.2f}. '
                             'FPS: {5: >8.2f}. ABS TIME [ms]: {6: >10.1f}.'.format(self.LOG_ORIGIN, self.cum_ep_tr, self.max_ep_tr, end_step, self.ep_returns_tr[-1], fps, elapsed_time))
        else:
            next_obs = obs
        # TODO: Remove this line.
        self.cur_ep_return_tr += reward

        self.cur_ep_decomp_return_tr.append(reward)
        self.cur_ep_decomp_opts_tr.append(option_id)

        return next_obs

    # TODO: Evaluation options agents requires special attention! Do later!
    def _eval_update(self, obs, reward, option_id, done, designated_option_id=None):
        self.cum_step_ev += 1
        self.cur_ep_step_ev += 1
        if done is True or self.cur_ep_step_ev > self.max_steps_per_ep:
            self.agent.on_new_episode()

            elapsed_time = time.perf_counter() - self.ep_start_t_ev
            fps = (self.cur_ep_step_ev - 1) / elapsed_time
            elapsed_time *= 1000

            self.ep_returns_ev.append(self.cur_ep_return_ev)
            self.cur_ep_return_ev = 0
            self.cur_ep_step_ev = 0

            next_obs = self._wrapped_env_reset(test=True)
            if self.render is True:
                ret = self._env_render(test=True)

            self.cur_ep_ev += 1
            self.ep_start_t_ev = time.perf_counter()

            self.logger.info('{0: <5} :: Episode {1: >4}/{2: <4} finished. FPS: {3: >8.2f}.'
                             ' ABS TIME [ms]: {4: >10.1f}.'.format(self.LOG_ORIGIN, self.cur_ep_ev, self.max_ep_ev, fps, elapsed_time))
        else:
            next_obs = obs
        self.cur_ep_return_ev += reward
        return next_obs

    def _eval_reset(self):
        self.cum_step_ev = 0
        self.cur_ep_ev = 0
        self.cur_ep_step_ev = 0
        self.cur_ep_return_ev = 0
        self.ep_returns_ev = []

    def train(self):
        self.LOG_ORIGIN = 'TRAIN'

        obs = self.init_obs_tr
        if self.render is True:
            ret = self._env_render(test=False)

        loss_summaries = []
        self.ep_start_t_tr = time.perf_counter()
        while self.cum_ep_tr < self.max_ep_tr and self.cum_step_tr < self.max_step_tr:
            action = self._select_action(obs=obs, test=False)
            next_obs, reward, option_id, done, _ = self._take_step(action=action, test=False)

            self.rb.add(obs=obs, action=action, next_obs=next_obs, reward=reward, done=done)
            obs = self._train_update(obs=next_obs, reward=reward, option_id=option_id, done=done)

            if self.cum_step_tr > self.warmup_steps:
                if self.cum_step_tr % self.agnt_updt_interval == 0:
                    batch = self.rb.sample(batch_size=self.batch_size)
                    loss_summaries = self.agent.update(
                        batch_obs=batch[0],
                        batch_act=batch[1],
                        batch_next_obs=batch[2],
                        batch_rew=batch[3],
                        batch_done=batch[4]
                    )
                    # if len(self.ep_returns_tr) > 0:
                    #     mean_ep_return = self._calc_mean_ep_return()
                    #     summary_id = 'mean_' + str(int(self.num_episode_returns)) + '_ep_return'
                    #     summary = (summary_id, 'scalar', mean_ep_return)
                    #     summaries.append(summary)

                if self.cum_step_tr % self.summary_interval == 0:
                    self.logger.info('{0: <5} :: Storing summaries @{1: <8} step.'.format(
                        self.LOG_ORIGIN, self.cum_step_tr))
                    self._store_summaries(loss_summaries=loss_summaries, test=False)

                if self.cum_step_tr % self.store_interval == 0:
                    self._store_model()

                if self.eval_agnt is True and self.cum_step_tr % self.agnt_ev_interval == 0:
                    cur_option_id = self.agent.get_cur_option_id()
                    self.evaluate()
                    self.agent.set_cur_option_id(cur_option_id)
                    self.LOG_ORIGIN = 'TRAIN'

    def evaluate(self):
        if self.eval_agnt is True:

            self.LOG_ORIGIN = 'EVAL'
            self.logger.info('{0: <5} :: Starting evaluation @{1: <8} step.'.format(self.LOG_ORIGIN, self.cum_step_tr))

            if self.separate_eval_per_option is True:
                options_to_test = self.cur_opt_performance_comparison[:self.max_opt_ev]
                designated_option_id = options_to_test[0]
                self.agent.set_cur_option_id(designated_option_id)
            else:
                options_to_test = [None]
                designated_option_id = None

            self.ep_start_t_ev = time.perf_counter()

            for opt_index in range(len(options_to_test) - 1):

                obs = self.init_obs_ev
                if self.render is True:
                    ret = self._env_render(test=True)

                while self.cur_ep_ev < self.max_ep_ev and self.cum_step_ev < self.max_step_ev:
                    action = self._select_action(obs=obs, test=True)
                    next_obs, reward, option_id, done, _ = self._take_step(action=action, test=True)
                    obs = self._eval_update(
                        obs=next_obs,
                        reward=reward,
                        option_id=option_id,
                        done=done,
                        designated_option_id=designated_option_id
                    )

                if len(self.ep_returns_ev) > 1 and self.stricter_exit_ev % self.max_steps_per_ep != 0:
                    self.ep_returns_ev = self.ep_returns_ev[:-1]

                if len(self.ep_returns_ev) > 0:
                    ep_reward = tf.convert_to_tensor(self.ep_returns_ev, dtype=tf.float32)
                    if len(self.ep_returns_ev) > 1:
                        mean_episode_reward = tf.squeeze(tf.math.reduce_mean(ep_reward))
                    else:
                        mean_episode_reward = tf.squeeze(ep_reward)
                    self.logger.info('{0: <5} :: {1: >4}/{2: <4} episodes FULLY finished. '
                                     'MEAN REWARD: {3: >10.2f}.'.format(self.LOG_ORIGIN, len(self.ep_returns_ev), self.max_ep_ev, float(mean_episode_reward.numpy())))
                    self._store_summaries(summaries=[('eval_mean_ep_return', 'scalar', mean_episode_reward)])
                else:
                    self.logger.warning('{0: <5} :: Stricter of the evaluation parameters max_steps and max_episodes,'
                                        ' combined with the max_step_per_episode do not permit even a single episode'
                                        ' evaluation. Evaluation results not stored!'.format(self.LOG_ORIGIN))
                # small reset
                self._eval_reset()
            # big reset?
        else:
            self.logger.warning('{0: <5} :: Evaluation environment not defined!'.format(self.LOG_ORIGIN))

    def _wrapped_env_reset(self, test):
        obs = self._env_reset(test=test)
        obs, option_id = self.agent.modify_observation(obs=obs)
        return obs, option_id

    @abstractmethod
    def _env_reset(self, test):
        # Sets the initial obs
        pass

    @abstractmethod
    def _env_act_sample(self):
        pass

    @abstractmethod
    def _env_act_select(self, obs, test):
        pass

    @abstractmethod
    def _env_action(self, action, test):
        pass

    @abstractmethod
    def _env_render(self, test):
        pass

    @abstractmethod
    def _copy_env(self):
        pass


class GymExperiment(Experiment):
    def __init__(self, env, agent, config_path, store_dir, eval_env=None, name=None, render_mode='default'):
        super().__init__(env, agent, config_path, store_dir, eval_env, name, render_mode)
        if self.render_mode == 'default':
            self.render_mode = 'human'
        else:
            self.render_mode = 'rgb_array'

    def _env_reset(self, test):
        if test is False:
            obs = self.env.reset()
        else:
            obs = self.eval_env.reset()
        return obs

    def _env_act_sample(self):
        action = self.env.action_space.sample()
        return action

    def _env_act_select(self, obs, test):
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        return self.agent.select_action(obs=tf.constant(obs), test=test)

    def _env_action(self, action, test):
        if isinstance(action.dtype, tf.DType):
            action = action.numpy()
        if test is False:
            observation, reward, done, info = self.env.step(action)
        else:
            observation, reward, done, info = self.eval_env.step(action)
        return observation, reward, done, info

    def _env_render(self, test):
        if test is False:
            ret_val = self.env.render(mode=self.render_mode)
        else:
            ret_val = self.eval_env.render(mode=self.render_mode)
        return ret_val

    def _copy_env(self):
        env_id = self.env.spec.id
        return gym.make(env_id)