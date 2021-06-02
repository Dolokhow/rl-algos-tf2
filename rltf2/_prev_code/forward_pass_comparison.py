import gym
import tensorflow as tf
import numpy as np

from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer

from rltf2.utils.third_party_wrappers import SACWrapper
from rltf2.utils.replay_buffer import CPPReplayBuffer
from rltf2.agents import sac

CONFIG_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/docs/config.yaml'
REPLAY_BUFFER_SIZE = 1e5
CUSTOM_RB = True
BATCH_DIM = 1
NUM_UPDATES = 10
TRY_TENSORS = True

tf.config.run_functions_eagerly(True)
np.random.seed(1234)
tf.random.set_seed(1234)

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--logdir', type=str,
                        default="/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/SAC/SAC_TF2RL")
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=3e6)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    shape = env.observation_space.shape

    tf2rl_policy = SACWrapper(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha)

    custom_policy = sac.SAC(
        action_shape=env.action_space.high.size,
        obs_shape=env.observation_space.shape,
        max_action=env.action_space.high[0],
        lr=3e-4,
        actor_units=[256, 256],
        critic_units=[256, 256],
        smooth_fact=0.005,
        temp=0.2,
        discount=0.99
    )

    rb = CPPReplayBuffer(
        storage_size=REPLAY_BUFFER_SIZE,
        params_dict={
            'obs_dtype': np.float32 if TRY_TENSORS is False else custom_policy.input_dtype,
            'act_dtype': np.float32 if TRY_TENSORS is False else custom_policy.action_dtype,
            'default_dtype': np.float32 if TRY_TENSORS is False else tf.float32,
            'env': {
                "obs": {"shape": env.observation_space.shape},
                "act": {"shape": env.action_space.shape},
                "rew": {},
                "next_obs": {"shape": env.observation_space.shape},
                "done": {}}
        }
    )

    # Copy parameters from TF2RL to our custom policy
    tf2rl_params = tf2rl_policy.get_weights()
    custom_policy.set_weights_from_dict(param_dict=tf2rl_params)
    init_percentage, init_net_percentages = custom_policy.compare_weights_from_dict(param_dict=tf2rl_params)
    comparisons = []

    for step in range(BATCH_DIM):
        dummy_obs = np.random.rand(1, 3).astype(np.float32)
        dummy_next_obs = np.random.rand(1, 3).astype(np.float32)
        dummy_actions = np.random.rand(1, 1).astype(np.float32)
        dummy_rewards = (np.random.rand(1, 1) * -1).astype(np.float32)
        dummy_dones = np.random.randint(0, 1, (1, 1)).astype(np.float32)
        rb.add(obs=dummy_obs, action=dummy_actions, reward=dummy_rewards, next_obs=dummy_next_obs, done=dummy_dones)

    for updt in range(NUM_UPDATES):
        dummy_obs_rb, dummy_actions_rb, dummy_next_obs_rb, dummy_rewards_rb, dummy_dones_rb = rb.sample(BATCH_DIM)

        debug_args = custom_policy.update_dummy(
            batch_obs=dummy_obs_rb,
            batch_act=dummy_actions_rb,
            batch_next_obs=dummy_next_obs_rb,
            batch_rew=dummy_rewards_rb,
            batch_done=dummy_dones_rb
        )
        q1_out = debug_args[0]
        q2_out = debug_args[1]
        next_v_targ = debug_args[2]
        v_out = debug_args[3]
        logp = debug_args[4]
        current_q = debug_args[5]
        sample_actions = debug_args[6]
        q1_out_cur = debug_args[7]
        q2_out_cur = debug_args[8]
        q1_loss = debug_args[9]
        q2_loss = debug_args[10]
        critic_loss = debug_args[11]
        actor_loss = debug_args[12]
        target_v = debug_args[13]
        target_q = debug_args[14]

        tf2rl_inputs = (dummy_obs_rb, dummy_actions_rb, dummy_next_obs_rb, dummy_rewards_rb, dummy_dones_rb)
        q1_out_tf2rl, q2_out_tf2rl, v_out_tf2rl, sample_actions_tf2rl, logp_tf2rl, q1_out_cur_tf2rl, \
            q2_out_cur_tf2rl, current_q_tf2rl, next_v_targ_tf2rl, q1_loss_tf2rl, q2_loss_tf2rl, critic_loss_tf2rl, \
            actor_loss_tf2rl, target_q_tf2rl, target_v_tf2rl = tf2rl_policy.test_update(inputs=tf2rl_inputs)

        side_by_side = {
            'inference': {
                'q1': [q1_out, q1_out_tf2rl],
                'q2': [q2_out, q2_out_tf2rl],
                'v': [v_out, v_out_tf2rl],
                'sampled_act': [sample_actions, sample_actions_tf2rl],
                'sample_q1': [q1_out_cur, q1_out_cur_tf2rl],
                'sample_q2': [q2_out_cur, q2_out_cur_tf2rl],
                'min_sample_q': [current_q, current_q_tf2rl],
                'logp': [logp, logp_tf2rl],
                'v_targ': [next_v_targ, next_v_targ_tf2rl]
            },
            'loss': {
                'q_target': [target_q, target_q_tf2rl],
                'q1_loss': [q1_loss, q1_loss_tf2rl],
                'q2_loss': [q2_loss, q2_loss_tf2rl],
                'v_target': [target_v, target_v_tf2rl],
                'critic_loss': [critic_loss, critic_loss_tf2rl],
                'actor_loss': [actor_loss, actor_loss_tf2rl]
            }
        }

        tf2rl_params = tf2rl_policy.get_weights()
        percentage, net_percentages = custom_policy.compare_weights_from_dict(param_dict=tf2rl_params)
        comparison = {
            'total_percentage': percentage,
            'vars': side_by_side,
            'net_percentages': net_percentages,
            'init_percentage': init_percentage,
            'initial_net_percentages': init_net_percentages
        }
        comparisons.append(comparison)
        print(percentage)

    print(0)


