from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf
import gym


class EnvInterface(ABC):
    def __init__(self, env):
        self.env = env
        # To be set within a subclass if exists
        self.step_limit = None

    @abstractmethod
    def get_obs_shape(self):
        pass

    @abstractmethod
    def get_action_shape(self):
        pass

    @abstractmethod
    def get_action_limit(self):
        pass

    @abstractmethod
    def env_reset(self):
        pass

    @abstractmethod
    def env_act_sample(self):
        pass

    @abstractmethod
    def env_action(self, action, episode_step):
        pass

    @abstractmethod
    def get_copy(self):
        pass


class GymInterface(EnvInterface):
    def __init__(self, env):
        super(GymInterface, self).__init__(env=env.env)
        self.step_limit = env.spec.max_episode_steps

    def get_obs_shape(self):
        return self.env.observation_space.shape

    def get_action_shape(self):
        return self.env.action_space.high.size

    def get_action_limit(self):
        return self.env.action_space.high[0]

    def env_reset(self):
        obs = self.env.reset()
        return obs

    def env_act_sample(self):
        action = self.env.action_space.sample()
        return action

    def env_action(self, action, episode_step):
        if isinstance(action.dtype, tf.DType):
            action = action.numpy()
        observation, reward, done, info = self.env.step(action)
        if bool(done) is True:
            markov_done = False if episode_step == self.step_limit else True
        else:
            markov_done = False
        return observation, reward, done, markov_done, info

    def get_copy(self):
        env_id = self.env.spec.id
        new_env = gym.make(env_id)
        return GymInterface(env=new_env)


class Renderer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def render_frame(self, env, option_id=None, total_reward=None, *args):
        pass


class RenderDummy(Renderer):
    def __init__(self):
        super(RenderDummy, self).__init__()

    def render_frame(self, env, option_id=None, total_reward=None, *args):
        pass


class GymRenderer(Renderer):
    def __init__(self, custom_render, store_dir=None):
        super(GymRenderer, self).__init__()
        if custom_render is True or store_dir is not None:
            self.render_mode = 'rgb_array'
        else:
            self.render_mode = 'human'
        self.store_dir = store_dir

    def render_frame(self, env, option_id=None, total_reward=None, *args):
        ret = env.render(mode=self.render_mode)
        if self.render_mode == 'rgb_array':
            # Implement custom rendering here using ret param!
            pass
