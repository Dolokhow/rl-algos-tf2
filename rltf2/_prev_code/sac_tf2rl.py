import gym
import tensorflow as tf
from tf2rl.algos.sac import SAC
from rltf2.utils.third_party_wrappers import SACWrapper
from rltf2.utils.experiment import GymExperiment
from rltf2.agents import sac
from tf2rl.experiments.trainer import Trainer


CUSTOM_TRAINER = False
CUSTOM_SAC = False
CONFIG_PATH = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/docs/config.yaml'

tf.config.run_functions_eagerly(True)
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

    policy = SACWrapper(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha
    )
    custom_policy = sac.SAC(
        action_shape=env.action_space.high.size,
        obs_shape=env.observation_space.shape,
        max_action=env.action_space.high[0],
        actor_units=(256, 256),
        critic_units=(256, 256),
        smooth_fact=5e-3,
        temp=args.alpha,
        discount=0.99,
        input_dtype=tf.float32
    )
    if CUSTOM_TRAINER is True:
        if CUSTOM_SAC is True:
            policy = custom_policy

        experiment = GymExperiment(
            env=env,
            agent=policy,
            store_dir='/results/prev_method/SAC',
            config_path=CONFIG_PATH,
            eval_env=None,
            name='SAC_Pendulum-v0'
        )
        experiment.train()
    else:
        trainer = Trainer(policy, env, args, test_env=test_env)
        trainer()
