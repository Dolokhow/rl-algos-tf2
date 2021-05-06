from abc import ABC, abstractmethod
import tensorflow as tf
from rltf2.utils.vector_ops import shape_expand_dim


class Agent(ABC, tf.keras.Model):
    def __init__(self, name, action_shape, obs_shape, discount=0.99, input_dtype=tf.float32, action_dtype=tf.float32):
        super(Agent, self).__init__(name=name)
        self.discount = discount

        # Add batch dimension
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.input_dtype = input_dtype
        self.action_dtype = action_dtype

    @staticmethod
    def _add_batch_dim(shape):
        return shape_expand_dim(shape=shape, axis=0)

    def set_weights_from_dict(self, param_dict):
        nn_counter = 0
        total_nns = len(self._inference_nets)

        for idx, weights in param_dict.items():
            for rl_nn in self._inference_nets:
                if rl_nn[0] == idx:
                    model = rl_nn[1]
                    model.soft_weight_update(
                        body_params=weights['body'],
                        out_layer_params=weights['out'],
                        smooth_fact=1.0)
                    nn_counter += 1
                    break

        if nn_counter != total_nns:
            raise ValueError('Mismatch between number of models to copy weights '
                             'to and provided weights in param_dict.')

    def compare_weights_from_dict(self, param_dict):
        nn_counter = 0
        percentage = 0
        percentages_per_net = []
        total_nns = len(self._inference_nets)

        for idx, weights in param_dict.items():
            for rl_nn in self._inference_nets:
                if rl_nn[0] == idx:
                    model = rl_nn[1]
                    percent = model.compare_weights(
                        body_params=weights['body'],
                        out_layer_params=weights['out']
                    )
                    percentage += percent
                    percentages_per_net.append((idx, percent))
                    nn_counter += 1
                    break

        if nn_counter != total_nns:
            raise ValueError('Mismatch between number of models to compare weights '
                             'to and provided weights in param_dict.')
        return percentage / total_nns, percentages_per_net

    def update(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done):
        summaries, _ = self._batch_inference(
            batch_obs=batch_obs,
            batch_act=batch_act,
            batch_next_obs=batch_next_obs,
            batch_rew=batch_rew,
            batch_done=batch_done,
            training=True
        )
        return summaries

    def update_dummy(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done):
        _, debug = self._batch_inference(
            batch_obs=batch_obs,
            batch_act=batch_act,
            batch_next_obs=batch_next_obs,
            batch_rew=batch_rew,
            batch_done=batch_done,
            training=False
        )
        return debug

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    @abstractmethod
    def _batch_inference(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done, training):
        pass

    @abstractmethod
    def select_action(self, obs, test=False):
        pass

    # Equivalent to Model.call(). Created as a separate entity because default arguments of Model.call() may
    # produce errors when trying to serialize the model with custom call arguments.
    @abstractmethod
    def forward_pass(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_forward_pass_input_spec(self):
        pass

    @abstractmethod
    def on_new_episode(self):
        pass

    def modify_observation(self, obs):
        return obs







