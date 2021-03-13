import tensorflow as tf
import tensorflow_probability as tfp


class MLPBody(tf.keras.layers.Layer):
    def __init__(self, layers_units, activation='relu', name='mlp_body'):
        super(MLPBody, self).__init__(name=name)
        self.layers = [tf.keras.layers.Dense(units=units, activation=activation) for units in layers_units]

    @tf.function
    def call(self, inputs, **kwargs):
        features = inputs
        for layer in self.layers:
            features = layer(features)
        return features


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, clip=True, squash=True, name='gaussian_sac'):
        super(GaussianLayer, self).__init__(name=name)
        if clip is True:
            self.LOG_STD_MIN = -20
            self.LOG_STD_MAX = 2
        else:
            self.LOG_STD_MIN = None
            self.LOG_STD_MAX = None
        self.EPS = 1e-6
        self.squash = squash
        self.train = False

        # linear mean layer
        self.mu_out = tf.keras.layers.Dense(units=feature_dim, activation=None, name='mu_out')
        # linear logarithm of the standard_dev layer
        self.log_std = tf.keras.layers.Dense(units=feature_dim, activation=None, name='sigma_out')

    def call(self, inputs, **kwargs):
        mu = self.mu_out(inputs)
        log_std = self.log_std(inputs)
        if self.LOG_STD_MIN is not None:
            log_std = tf.clip_by_value(log_std, clip_value_min=self.LOG_STD_MIN, clip_value_max=self.LOG_STD_MAX)
        # Diagonal variance matrix is exponentiated as output standard deviation layer predicts log values
        cur_distr = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_std))
        return cur_distr
