from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf
import gym


class EnvInterface(ABC):
    def __init__(self, env):
        self.env = env

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
        # Sets the initial obs
        pass

    @abstractmethod
    def env_act_sample(self):
        pass

    @abstractmethod
    def env_action(self, action):
        pass

    @abstractmethod
    def get_copy(self):
        pass


class GymInterface(EnvInterface):
    def __init__(self, env):
        super(GymInterface, self).__init__(env=env)

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

    def env_action(self, action):
        if isinstance(action.dtype, tf.DType):
            action = action.numpy()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def get_copy(self):
        env_id = self.env.spec.id
        new_env = gym.make(env_id)
        return GymInterface(env=new_env)


class Renderer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def render_frame(self, env):
        pass


class RenderDummy(Renderer):
    def __init__(self):
        super(RenderDummy, self).__init__()

    def render_frame(self, env):
        pass


class GymRenderer(Renderer):
    def __init__(self, custom_render):
        super(GymRenderer, self).__init__()
        if custom_render is True:
            self.render_mode = 'rgb_array'
        else:
            self.render_mode = 'human'

    def render_frame(self, env):
        ret = env.render(mode=self.render_mode)
        if self.render_mode == 'rgb_array':
            # Implement custom rendering here using ret param!
            pass
