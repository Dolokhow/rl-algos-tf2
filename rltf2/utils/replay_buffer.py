import tensorflow as tf
import random
from abc import ABC, abstractmethod
import cpprb
import numpy as np


def get_replay_buffer(rb_type, storage_size, params_dict):
    if rb_type == 'custom':
        return ReplayBuffer(
            storage_size=storage_size,
            obs_dtype=params_dict['obs_dtype'],
            act_dtype=params_dict['act_dtype'],
            default_dtype=params_dict['default_dtype']
        )
    elif rb_type == 'cpprb':
        return CPPReplayBuffer(
            storage_size=storage_size,
            params_dict=params_dict
        )
    else:
        return CPPReplayBuffer(
            storage_size=storage_size,
            params_dict=params_dict
        )


class ReplayBufferProto(ABC):
    def __init__(self, obs_dtype, act_dtype, default_dtype):
        self.obs_dtype = obs_dtype
        self.act_dtype = act_dtype
        self.default_dtype = default_dtype
    
    @abstractmethod
    def add(self, obs, action, next_obs, reward, done) -> bool:
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass


class CPPReplayBuffer(ReplayBufferProto):
    def __init__(self, storage_size, params_dict):
        super(CPPReplayBuffer, self).__init__(
            obs_dtype=params_dict['obs_dtype'],
            act_dtype=params_dict['act_dtype'],
            default_dtype=params_dict['default_dtype']
        )
        self.rb = cpprb.ReplayBuffer(storage_size, params_dict['env'])

    def add(self, obs, action, next_obs, reward, done):
        self.rb.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
        return True

    def sample(self, batch_size):
        sample = self.rb.sample(batch_size)
        obss = sample['obs']
        actions = sample['act']
        next_obss = sample['next_obs']
        rewards = sample['rew']
        dones = sample['done']

        if isinstance(self.obs_dtype, tf.DType):
            obss = tf.convert_to_tensor(obss, dtype=self.obs_dtype)
            next_obss = tf.convert_to_tensor(next_obss, dtype=self.obs_dtype)
        if isinstance(self.act_dtype, tf.DType):
            actions = tf.convert_to_tensor(actions, dtype=self.act_dtype)
        if isinstance(self.default_dtype, tf.DType):
            rewards = tf.convert_to_tensor(rewards, dtype=self.default_dtype)
        if isinstance(self.default_dtype, tf.DType):
            dones = tf.convert_to_tensor(dones, dtype=self.default_dtype)

        return obss, actions, next_obss, rewards, dones


# Should handle concurrent logic in case of multiple agents accessing it
class ReplayBuffer(ReplayBufferProto):
    def __init__(self, storage_size, obs_dtype, act_dtype, default_dtype):
        super(ReplayBuffer, self).__init__(
            obs_dtype=obs_dtype,
            act_dtype=act_dtype,
            default_dtype=default_dtype
        )
        self._data = []
        self.max_len = int(storage_size)

        self._next_id = 0
        self._terminals = set([])
        self._write_flg = True
        random.seed(123)

    @staticmethod
    def _preprocess(var, dtype):
        if hasattr(var, 'shape') and len(var.shape) > 1 and var.shape[1] == 1:
            if tf.is_tensor(var):
                var = tf.squeeze(var, axis=1)
            else:
                var = np.squeeze(var, axis=1)

        if hasattr(var, 'dtype') and var.dtype is not dtype:
            if tf.is_tensor(var):
                var = (tf.make_ndarray(var)).astype(dtype)
            else:
                var = tf.convert_to_tensor(var, dtype=dtype)

        if not hasattr(var, 'dtype') and isinstance(dtype, tf.DType):
            var = tf.convert_to_tensor(var, dtype=dtype)

        return var

    def _add(self,  obs, action, next_obs, reward, done):

        obs = self._preprocess(var=obs, dtype=self.obs_dtype)
        action = self._preprocess(var=action, dtype=self.act_dtype)
        next_obs = self._preprocess(var=next_obs, dtype=self.obs_dtype)
        reward = self._preprocess(var=reward, dtype=self.default_dtype)
        done = self._preprocess(var=done, dtype=self.default_dtype)

        if len(self._data) < self.max_len:
            self._data.append((obs, action, next_obs, reward, done))
        else:
            self._data[self._next_id] = (obs, action, next_obs, reward, done)

        self._terminals.discard(self._next_id)
        if done is True:
            self._terminals.add(self._next_id)

        self._next_id = (self._next_id + 1) % self.max_len

    def add(self, obs, action, next_obs, reward, done):
        if self._write_flg is True:
            self._write_flg = False
            self._add(obs, action, next_obs, reward, done)
            self._write_flg = True
        return self._write_flg

    def _sample(self, batch_size):
        cap = len(self._data)
        idxs = [random.randint(0, cap-1) for _ in range(0, batch_size)]
        obss, actions, next_obss, rewards, dones = [], [], [], [], []

        for idx in idxs:
            data = self._data[idx]
            obss.append(data[0])
            actions.append(data[1])
            next_obss.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])

        obss = tf.convert_to_tensor(value=obss, dtype=self.obs_dtype)
        actions = tf.convert_to_tensor(value=actions, dtype=self.act_dtype)
        next_obss = tf.convert_to_tensor(value=next_obss, dtype=self.act_dtype)
        rewards = tf.convert_to_tensor(value=rewards, dtype=self.default_dtype)
        dones = tf.convert_to_tensor(value=dones, dtype=self.default_dtype)

        return obss, actions, next_obss, rewards, dones

    # TODO: Implement this method
    def _sample_sequential(self, batch_size):
        return None, None, None, None, None

    def sample(self, batch_size, sequence=False):
        self._write_flg = False
        if sequence is False:
            obss, actions, next_obss, rewards, dones = self._sample(batch_size=batch_size)
        else:
            obss, actions, next_obss, rewards, dones = self._sample_sequential(batch_size=batch_size)
        self._write_flg = True
        return obss, actions, next_obss, rewards, dones

