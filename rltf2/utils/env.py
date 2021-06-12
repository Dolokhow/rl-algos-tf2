from abc import abstractmethod, ABC
import tensorflow as tf
import gym
import os
import cv2
from rltf2.utils.visualization_utils import create_video, paint_text


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
    def __init__(self, name=None, store_dir=None, txt_color=(255, 255, 255)):
        if name is None:
            self.name = ''
        else:
            self.name = name
        self.store_dir = store_dir
        self.vid_cap: cv2.VideoWriter = None

        self.id_txt = 'Opt ID: '
        self.rew_txt = 'Returns: '
        self.txt_color = txt_color

    @abstractmethod
    def render_frame(self, env, option_id=None, total_reward=None, *args):
        pass

    def _create_video(self, width, height, opt_tag=''):
        if self.vid_cap is None:
            video_name = self.name + '_opt_' + opt_tag
            videos_in_dir = [f for f in os.listdir(self.store_dir) if
                             os.path.isfile(os.path.join(self.store_dir, f)) and f.endswith('.mp4') and video_name in f]
            video_name += '_' + str(len(videos_in_dir))
            path = os.path.join(self.store_dir, video_name)
            params = (width, height, 30)
            self.vid_cap = create_video(path=path, code='mp4v', params=params, extension='.mp4')

    def end_video(self):
        if self.vid_cap is not None:
            self.vid_cap.release()
            self.vid_cap = None


class RenderDummy(Renderer):
    def __init__(self):
        super(RenderDummy, self).__init__()

    def render_frame(self, env, option_id=None, total_reward=None, *args):
        pass


class GymRenderer(Renderer):
    def __init__(self, custom_render, name=None, store_dir=None):
        super(GymRenderer, self).__init__(name=name, store_dir=store_dir)
        if custom_render is True:
            self.render_mode = 'rgb_array'
        else:
            self.render_mode = 'human'

    # Video storing is supported only if custom rendering is set to True, or rather
    # if self._render_mode == 'rgb_array'.
    # TODO: Implement video storing functionality independent of the render mode.
    def render_frame(self, env, option_id=None, total_reward=None, *args):
        ret = env.render(mode=self.render_mode)
        if self.render_mode == 'rgb_array':
            frame_w = ret.shape[1]
            frame_h = ret.shape[0]
            # horizontal_pos = max(int(0.5 * frame_w), max(10, frame_w - 400))
            position = (10, 30)
            ret = paint_text(
                frame=ret,
                text=self.id_txt + str(option_id),
                color=self.txt_color,
                position=position
            )
            position = (10, 60)
            ret = paint_text(
                frame=ret,
                text=self.rew_txt + str(int(total_reward)),
                color=self.txt_color,
                position=position
            )
            if self.store_dir is not None:
                # _create_video will check if video is alrady created
                self._create_video(width=frame_w, height=frame_h, opt_tag=str(option_id))
                self.vid_cap.write(ret)

            # Paint necessary information onto the frame
            cv2.imshow(self.name, ret)
            cv2.waitKey(5)
