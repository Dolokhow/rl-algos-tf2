import tensorflow as tf
from abc import ABC
from rltf2.core.rl_layers import GaussianLayer


class RLNet(tf.keras.Model, ABC):
    def __init__(self, body, input_dim, lr=None, optimizer=None, name=None):
        super(RLNet, self).__init__(name=name)
        self.body = body
        self._input_dim = input_dim

        if optimizer is None:
            assert lr is not None
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer = optimizer

        # To be set as within a subclass
        self.out_layer = None

    def soft_weight_update(self, body_params, out_layer_params, smooth_fact):
        def update_op(target_var, src_var):
            if smooth_fact == 1.0:
                target_var.assign(src_var)
            else:
                target_var.assign(smooth_fact * src_var + (1.0 - smooth_fact) * target_var)

        for target, source in zip(self.body.weights, body_params):
            update_op(target_var=target, src_var=source)
        for target, source in zip(self.out_layer.weights, out_layer_params):
            update_op(target_var=target, src_var=source)

    # TODO: Test this method.
    def compare_weights(self, body_params, out_layer_params):
        def compare_op(target_var, src_var):
            equality = tf.math.equal(target_var, src_var)
            percentage = tf.reduce_sum(tf.cast(equality, tf.float32))/tf.cast(tf.size(equality), tf.float32)
            return percentage

        ind = 0
        equal_percentage = 0
        for target, source in zip(self.body.weights, body_params):
            equal_percentage += compare_op(target_var=target, src_var=source)
            ind += 1
        for target, source in zip(self.out_layer.weights, out_layer_params):
            equal_percentage += compare_op(target_var=target, src_var=source)
            ind += 1
        total_percentage = equal_percentage / ind
        return total_percentage

    def get_config(self):
        pass


class StochasticNormalNet(RLNet):
    def __init__(self, body, input_dim, act_feature_dim, max_action,
                 lr=None, optimizer=None, squash=True, eps=1e-6, name='policy_net'):
        super(StochasticNormalNet, self).__init__(body=body, input_dim=input_dim, lr=lr,
                                                  optimizer=optimizer, name=name)
        self.out_layer = GaussianLayer(feature_dim=act_feature_dim)
        self.squash = squash
        self.eps = eps
        self.max_action = max_action
        self.build(input_shape=self._input_dim)

    @tf.function
    def call(self, inputs, training=True, **kwargs):
        logits = self.body(inputs=inputs)
        # Run inference through Gaussian Layer & get mean and standard deviation
        # Create a Gaussian Distribution object using mean and standard deviation
        cur_distr = self.out_layer(inputs=logits)
        # Sample the distribution; At test time take means across batch actions as optimal actions
        # at train time actions are stochastic and sample the distribution
        if not training:
            raw_smpl = cur_distr.mean()
        else:
            # TODO: Find out if it does re-parametrization by default! Explicit re-parametrization missing!
            raw_smpl = cur_distr.sample()
        # Compute logarithm of the policy
        log_policy = cur_distr.log_prob(raw_smpl)

        # SAC Appendix C -- Enforcing Action Bounds
        if self.squash:
            action_sample = tf.tanh(raw_smpl)
            diff = tf.reduce_sum(tf.math.log(1 - action_sample ** 2 + self.eps), axis=1)
            log_policy -= diff
        else:
            action_sample = raw_smpl

        # Bounding an action based on the environment definitions
        action_sample = action_sample * self.max_action
        return action_sample, log_policy


class VNet(RLNet):
    def __init__(self, body, input_dim, lr=None, optimizer=None, name='value_net'):
        super(VNet, self).__init__(body=body, input_dim=input_dim, lr=lr,
                                   optimizer=optimizer, name=name)
        self.out_layer = tf.keras.layers.Dense(units=1, activation='linear', name='value_out')
        self.build(input_shape=self._input_dim)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        logits = self.body(inputs=inputs)
        cur_output = self.out_layer(inputs=logits)
        return tf.squeeze(cur_output, axis=1)
