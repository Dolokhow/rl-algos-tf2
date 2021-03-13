from abc import ABC, abstractmethod
import tensorflow as tf
from rltf2.core.rl_nn import RLNet
from typing import List


class Agent(ABC, tf.keras.Model):
    def __init__(self, name, discount=0.99, input_dtype=tf.float32, action_dtype=tf.float32):
        super(Agent, self).__init__(name=name)
        self.discount = discount
        self.input_dtype = input_dtype
        self.action_dtype = action_dtype

        # TO BE SET WITHIN A SUBCLASS
        # For serialization used for inference with frozen model
        self.inference_model_dict = None
        # List of actual neural networks that run inference either during training, or evaluation.
        # To be set within a subclass.
        self._rl_nns: List[(str, RLNet)] = []

    def set_weights_from_dict(self, param_dict):
        nn_counter = 0
        total_nns = len(self._rl_nns)

        for idx, weights in param_dict.items():
            for rl_nn in self._rl_nns:
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
        total_nns = len(self._rl_nns)

        for idx, weights in param_dict.items():
            for rl_nn in self._rl_nns:
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

    @abstractmethod
    def select_action(self, obs, test=False):
        pass

    @abstractmethod
    def _batch_inference(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done, training):
        pass

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

    def get_config(self):
        pass




