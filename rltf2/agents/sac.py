import tensorflow as tf
from rltf2.core.rl_nn import StochasticNormalNet, VNet
from rltf2.core.rl_layers import MLPBody
from rltf2.agents.agent import Agent


class SAC(Agent):
    def __init__(self, action_shape, obs_shape, max_action=1., lr=3e-4, actor_units=(256, 256),
                 critic_units=(256, 256), smooth_fact=5e-3, temp=.2, discount=0.99, input_dtype=tf.float32):
        super(SAC, self).__init__(name='SAC',
                                  action_shape=action_shape,
                                  obs_shape=obs_shape,
                                  discount=discount,
                                  input_dtype=input_dtype,
                                  action_dtype=tf.float32
                                  )
        self.temp = temp
        self.smooth_fact = smooth_fact

        obs_shape_lst = list(self.obs_shape)
        obs_shape_lst[-1] += self.action_shape[-1]
        critic_q_input_shape = tuple(obs_shape_lst)

        self.critic_q1 = VNet(
            body=MLPBody(layers_units=critic_units, name='Q1_MLP'),
            input_dim=critic_q_input_shape,
            name='Q1_Critic',
            lr=lr,
            optimizer=None
        )
        self.critic_q2 = VNet(
            body=MLPBody(layers_units=critic_units, name='Q2_MLP'),
            input_dim=critic_q_input_shape,
            name='Q2_Critic',
            lr=lr,
            optimizer=None
        )
        self.critic_v = VNet(
            body=MLPBody(layers_units=critic_units, name='V_MLP'),
            input_dim=self.obs_shape,
            name='V_Critic',
            lr=lr,
            optimizer=None
        )
        self.critic_v_targ = VNet(
            body=MLPBody(layers_units=critic_units, name='V_MLP'),
            input_dim=self.obs_shape,
            name='V_Critic_Target',
            lr=lr,
            optimizer=None
        )
        self.critic_v_targ.soft_weight_update(
            body_params=self.critic_v.body.weights,
            out_layer_params=self.critic_v.out_layer.weights,
            smooth_fact=1.0
        )
        self.actor = StochasticNormalNet(
            body=MLPBody(layers_units=actor_units, name='Act_MLP'),
            input_dim=self.obs_shape,
            name='Stochastic_Actor',
            act_feature_dim=action_shape,
            max_action=max_action,
            lr=lr,
            optimizer=None
        )
        self.inference_model_dict = {self.name + '_actor_': self.actor}
        self._rl_nns = [('q1', self.critic_q1), ('q2', self.critic_q2),
                        ('v', self.critic_v), ('actor', self.actor), ('v_targ', self.critic_v_targ)]

    def get_serialization_dict(self):
        return {self.name + '_actor': self.actor}

    @tf.function
    def select_action(self, obs, test=False):
        action, _ = self.actor(obs, training=not test)
        return tf.squeeze(action, axis=1)

    @tf.function
    def forward_pass(self, batch_obs, batch_act, batch_next_obs, training=True):
        q_critic_inputs = tf.concat((batch_obs, batch_act), axis=1)
        q1_out = self.critic_q1(q_critic_inputs)
        q2_out = self.critic_q2(q_critic_inputs)
        next_v_targ = self.critic_v_targ(batch_next_obs)

        v_out = self.critic_v(batch_obs)
        sample_actions, logp = self.actor(batch_obs, training=training)
        q_critic_inputs_s = tf.concat((batch_obs, sample_actions), axis=1)
        q1_out_smpl = self.critic_q1(q_critic_inputs_s)
        q2_out_smpl = self.critic_q2(q_critic_inputs_s)
        min_q_smpl = tf.minimum(q1_out_smpl, q2_out_smpl)
        return q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, q1_out_smpl, q2_out_smpl

    def get_forward_pass_input_spec(self):
        batch_obs = tf.TensorSpec(shape=self.obs_shape, dtype=self.input_dtype, name="batch_obs")
        batch_act = tf.TensorSpec(shape=self.action_shape, dtype=self.action_dtype, name="batch_act")
        batch_next_obs = tf.TensorSpec(shape=self.obs_shape, dtype=self.input_dtype, name="batch_next_obs")
        kwargs = {
            'batch_obs': batch_obs,
            'batch_act': batch_act,
            'batch_next_obs': batch_next_obs,
            'training': True
        }
        return kwargs

    @tf.function
    def _batch_inference(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done, training):
        if len(batch_rew.shape) > 1 and batch_rew.shape[1] == 1:
            batch_rew = tf.squeeze(batch_rew, axis=1)
        if len(batch_done.shape) > 1 and batch_done.shape[1] == 1:
            batch_done = tf.squeeze(batch_done, axis=1)

        batch_not_done = 1. - tf.cast(batch_done, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, \
                q1_out_smpl, q2_out_smpl = self.forward_pass(
                    batch_obs=batch_obs,
                    batch_act=batch_act,
                    batch_next_obs=batch_next_obs,
                    training=training
                )

            # Equation (7)
            q_targ = tf.stop_gradient(batch_rew + batch_not_done * self.discount * next_v_targ)
            q1_loss = tf.reduce_mean(tf.math.square(q1_out - q_targ))
            q2_loss = tf.reduce_mean(tf.math.square(q2_out - q_targ))

            # Equation (5)
            v_targ = tf.stop_gradient(min_q_smpl - self.temp * logp)
            critic_loss = tf.reduce_mean(tf.math.square(v_out - v_targ))

            # Equation (12)
            actor_loss = tf.reduce_mean(self.temp * logp - min_q_smpl)

        summary_args = []
        debug_args = [q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, q1_out_smpl, q2_out_smpl]
        for rl_nn, loss in zip(self._rl_nns[:-1], [q1_loss, q2_loss, critic_loss, actor_loss]):
            model = rl_nn[1]
            summary_args.append((model.name + "_loss", "scalar", loss))
            debug_args.append(loss)
            grad = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        debug_args.extend([v_targ, q_targ])

        self.critic_v_targ.soft_weight_update(
            body_params=self.critic_v.body.weights,
            out_layer_params=self.critic_v.out_layer.weights,
            smooth_fact=self.smooth_fact
        )
        return summary_args, debug_args

