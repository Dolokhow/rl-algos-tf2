import tensorflow as tf
import numpy as np
from rltf2.agents.sac import SAC
from rltf2.core.nn_layers import MLPBody
from rltf2.core.rl_nets import GPClassifier
from rltf2.utils.vector_ops import shape_expand_axis, split_vector, force_merge_vectors, \
    broadcast_1d_row_vector, shape_expand_dim


class DIAYN(SAC):
    def __init__(self, action_shape, obs_shape, num_options=20, max_action=1., lr=3e-4, actor_units=(256, 256),
                 critic_units=(256, 256), discriminator_units=(100, 100), smooth_fact=5e-3, temp=.2, discount=0.99,
                 input_dtype=tf.float32):
        # Constructs SAC which is a basis for DIYAN. Observation that SAC receives is [env_obs, option_one_hot].
        # SAC QCritic will thus receive [env_obs, option_one_hot, action].
        # TODO: See how this relates to original DIAYN dimension handling as seen here:
        # https://github.com/haarnoja/sac/blob/master/examples/mujoco_all_diayn.py; 176-181
        # uses np.hstack which does the same thing for 1D arrays. Quote from the paper:
        # To pass skill z to the Q function, value function, and policy, we simply concatenate z to the current state st

        # Add one-hot option identifier (z) to the observation dimension
        modified_obs_shape = shape_expand_axis(shape=obs_shape, axis=-1, size=num_options)
        super(DIAYN, self).__init__(name='DIAYN',
                                    action_shape=action_shape,
                                    obs_shape=modified_obs_shape,
                                    max_action=max_action,
                                    lr=lr,
                                    actor_units=actor_units,
                                    critic_units=critic_units,
                                    smooth_fact=smooth_fact,
                                    temp=temp,
                                    discount=discount,
                                    input_dtype=input_dtype
                                    )
        self.env_obs_shape = obs_shape
        self._num_options = num_options
        self._cur_option_id = self._sample_option()

        self.discriminator = GPClassifier(
            body=MLPBody(layers_units=discriminator_units, name='Discriminator_MLP'),
            input_dim=shape_expand_dim(obs_shape, axis=0),
            num_classes=num_options,
            raw_logits_out=True,
            name='Discriminator',
            lr=lr,
            optimizer=None
        )
        # Insert the learned discriminator model in the second to last position,
        # as last is reserved for non learned model
        self._learned_nets.append(('opt_disrim', self.discriminator))

    # Option handling

    def _sample_option(self):
        return np.random.randint(0, self._num_options)

    def _one_hot(self, option_id):
        one_hot = np.zeros(self._num_options)
        one_hot[option_id] = 1
        return one_hot

    def modify_observation(self, obs):
        option = self._one_hot(option_id=self._cur_option_id)
        # This usage of force_merge_vectors is OK, but be careful in general.
        modified_obs = force_merge_vectors(t1=obs, t2=option, axis=1, pref_vtype=None)
        return modified_obs

    def on_new_episode(self):
        self._cur_option_id = self._sample_option()

    @tf.function
    def forward_pass(self, batch_obs, batch_act, batch_next_obs, original_batch_next_obs, training=True):
        q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, q1_out_smpl, q2_out_smpl = \
            super(DIAYN, self).forward_pass(
                batch_obs=batch_obs,
                batch_act=batch_act,
                batch_next_obs=batch_next_obs,
                training=training
            )
        discriminator_out = self.discriminator(original_batch_next_obs)
        return q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, \
            q1_out_smpl, q2_out_smpl, discriminator_out

    @tf.function
    def _batch_inference(self, batch_obs, batch_act, batch_next_obs, batch_rew, batch_done, training):

        if len(batch_done.shape) > 1 and batch_done.shape[1] == 1:
            batch_done = tf.squeeze(batch_done, axis=1)

        batch_not_done = 1. - tf.cast(batch_done, tf.float32)
        original_batch_next_obs, gt_batch_options = split_vector(
            t=batch_next_obs,
            index=-self._num_options,
            axis=1
        )

        with tf.GradientTape(persistent=True) as tape:
            q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, \
                q1_out_smpl, q2_out_smpl, discriminator_out = self.forward_pass(
                    batch_obs=batch_obs,
                    batch_act=batch_act,
                    batch_next_obs=batch_next_obs,
                    original_batch_next_obs=original_batch_next_obs,
                    training=training
                )

            # Augmented DIAYN option based reward: See first term of equation (3) in DIAYN
            # log p(z) is omitted since we use categorical uniform distribution for sampling options
            discr_log = tf.nn.softmax_cross_entropy_with_logits(
                labels=gt_batch_options,
                logits=discriminator_out
            )

            # Equation (7) SAC, reward from environment substituted by option based reward from DIAYN
            q_targ = tf.stop_gradient(-1 * discr_log + batch_not_done * self.discount * next_v_targ)
            q1_loss = tf.reduce_mean(tf.math.square(q1_out - q_targ))
            q2_loss = tf.reduce_mean(tf.math.square(q2_out - q_targ))

            # Equation (5)
            v_targ = tf.stop_gradient(min_q_smpl - self.temp * logp)
            critic_loss = tf.reduce_mean(tf.math.square(v_out - v_targ))

            # Equation (12)
            actor_loss = tf.reduce_mean(self.temp * logp - min_q_smpl)

            # DIAYN discriminator loss
            discriminator_loss = tf.reduce_mean(discr_log)

        summary_args = []
        debug_args = [q1_out, q2_out, next_v_targ, v_out, logp, min_q_smpl, sample_actions, q1_out_smpl, q2_out_smpl]
        for rl_nn, loss in zip(self._learned_nets, [q1_loss, q2_loss, critic_loss, actor_loss, discriminator_loss]):
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


