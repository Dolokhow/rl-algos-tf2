from tf2rl.algos.sac import SAC
from rltf2.agents.agent import Agent
from tf2rl.misc.target_update_ops import update_target_variables
import tensorflow as tf
from abc import ABC, abstractmethod


class AlgoWrapper(ABC):

    @abstractmethod
    def test_input(self, inputs, *args, **kwargs):
        pass

    @abstractmethod
    def get_weights(self):
        pass


class SACWrapper(SAC, AlgoWrapper):
    def __init__(self, state_shape, action_dim, name="SAC", max_action=1., lr=3e-4, lr_alpha=3e-4,
                 actor_units=(256, 256), critic_units=(256, 256), tau=5e-3, alpha=.2, auto_alpha=False,
                 n_warmup=int(1e4), memory_capacity=int(1e6), replay_buffer=None, input_dtype=tf.float32, **kwargs):
        super(SACWrapper, self).__init__(state_shape=state_shape, action_dim=action_dim, name=name,
                                         max_action=max_action, lr=lr, lr_alpha=lr_alpha, actor_units=actor_units,
                                         critic_units=critic_units, tau=tau, alpha=alpha, auto_alpha=auto_alpha,
                                         n_warmup=n_warmup, memory_capacity=memory_capacity, **kwargs)
        self.rb = replay_buffer
        self.input_dtype = input_dtype

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def select_action(self, obs, test=False):
        obs = tf.squeeze(obs).numpy()
        return self.get_action(state=obs, test=test)

    def test_input(self, inputs, *args, **kwargs):
        states = inputs[0]
        actions = inputs[1]
        next_states = inputs[2]

        current_q1 = self.qf1(states, actions)
        current_q2 = self.qf2(states, actions)
        next_v_target = self.vf_target(next_states)
        current_v = self.vf(states)
        sample_actions, logp = self.actor(states, test=True)  # Resample actions to update V
        current_q1_next = self.qf1(states, sample_actions)
        current_q2_next = self.qf2(states, sample_actions)
        current_min_q_next = tf.minimum(current_q1_next, current_q2_next)
        return current_q1, current_q2, current_v, sample_actions, logp, current_q1_next, \
               current_q2_next, current_min_q_next, next_v_target

    def test_update(self, inputs, *args, **kwargs):
        states = inputs[0]
        actions = inputs[1]
        next_states = inputs[2]
        rewards = inputs[3]
        dones = inputs[4]

        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.cast(tf.squeeze(rewards, axis=1), tf.float32)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)
                next_v_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * next_v_target)

                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(7)

                # Compute loss of critic V
                current_v = self.vf(states)

                sample_actions, logp = self.actor(states, test=True)  # Resample actions to update V
                initial_q1_out = current_q1
                current_q1 = self.qf1(states, sample_actions)
                initial_q2_out = current_q2
                current_q2 = self.qf2(states, sample_actions)
                current_min_q = tf.minimum(current_q1, current_q2)

                target_v = tf.stop_gradient(current_min_q - self.alpha * logp)
                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(td_errors ** 2)  # Eq.(5)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(
                zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(
                self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

            del tape
        return initial_q1_out, initial_q2_out, current_v, sample_actions, logp, current_q1, current_q2, current_min_q, \
               next_v_target, td_loss_q1, td_loss_q2, td_loss_v, policy_loss, target_q, target_v

    # Gets list of all neural network weights
    def get_weights(self):
        ret_dict = {}
        q1_body = []
        for layer in self.qf1.base_layers:
            q1_body.extend(layer.trainable_variables)
        q1_out_layer = self.qf1.out_layer.trainable_variables
        ret_dict['q1'] = {
            'body': q1_body,
            'out': q1_out_layer
        }

        q2_body = []
        for layer in self.qf2.base_layers:
            q2_body.extend(layer.trainable_variables)
        q2_out_layer = self.qf2.out_layer.trainable_variables
        ret_dict['q2'] = {
            'body': q2_body,
            'out': q2_out_layer
        }

        v_body = []
        for layer in self.vf.base_layers:
            v_body.extend(layer.trainable_variables)
        v_out_layer = self.vf.out_layer.trainable_variables
        ret_dict['v'] = {
            'body': v_body,
            'out': v_out_layer
        }

        actor_body = []
        for layer in self.actor.base_layers:
            actor_body.extend(layer.trainable_variables)
        actor_out = self.actor.out_mean.trainable_variables
        actor_out.extend(self.actor.out_logstd.trainable_variables)
        ret_dict['actor'] = {
            'body': actor_body,
            'out': actor_out
        }

        vt_body = []
        for layer in self.vf_target.base_layers:
            vt_body.extend(layer.trainable_variables)
        vt_out_layer = self.vf_target.out_layer.trainable_variables
        ret_dict['v_targ'] = {
            'body': vt_body,
            'out': vt_out_layer
        }

        return ret_dict

    def update(self):
        obss, actions, next_obss, rewards, dones = self.rb.sample(batch_size=self.batch_size)
        self.train(
            states=obss.numpy(),
            actions=tf.expand_dims(actions, axis=1).numpy(),
            next_states=next_obss.numpy(),
            rewards=tf.expand_dims(rewards, axis=1).numpy(),
            dones=tf.expand_dims(dones, axis=1).numpy()
        )
        return []








