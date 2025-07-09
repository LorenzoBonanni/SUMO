import os
import datetime
import random
import importlib
import pandas as pd
import d4rl
import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.stats import linregress
from algo.sac import SACPolicy
from common.buffer import ReplayBuffer, RolloutBuffer
from common.logger import Logger
from common.util import get_args, set_device_and_logger
from models.policy_models import Critic, ActorProb, MLP, DiagGaussian
from models.transition_model import TransitionModel


SAMPLE_SIZE = 100
ROLLOUT_LENGTH = 100

def plot_uncertainty_vs_distance(dist, uncertainty, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(dist, uncertainty, alpha=0.5, label='Data points')
    slope, intercept, r_value, p_value, std_err = linregress(dist, uncertainty)
    x_vals = np.linspace(dist.min(), dist.max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', label=f'Fit line (r={r_value:.2f})')
    plt.xlabel('Distance')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty vs Distance')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_l2_distance(predicted, true):
    distances = np.sqrt(np.sum((predicted - true) ** 2, axis=1))
    return distances


def compute_spearman_rank(distances, penalties):
    assert distances.shape == penalties.shape
    n = len(distances)

    def compute_rank(arr):
        ranks = np.empty_like(arr)
        ranks[np.argsort(arr)] = np.arange(n)
        return ranks

    rank_dist = compute_rank(distances)
    rank_pen = compute_rank(penalties)

    d = rank_dist - rank_pen
    spearman = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))

    return spearman


def compute_pearson_rank(distances, penalties):
    return np.corrcoef(distances, penalties)[0, 1]


def rollout_data(dynamics_model, policy, offline_buffer, online_buffer, env, sample_size=100, rollout_length=100):
    observations = offline_buffer.sample(sample_size)["observations"]

    for _ in range(rollout_length):
        actions = policy.sample_action(observations)
        next_obs, rewards, terminals, infos = dynamics_model.predict(observations, actions, penalty_coeff=1.0)
        online_buffer.add_batch(observations, actions, next_obs, np.zeros_like(next_obs), infos["penalty"])

        non_terminal = (~terminals).flatten()
        if non_terminal.sum() == 0:
            break
        observations = next_obs[non_terminal]

    for idx in range(online_buffer.size):
        env.state = online_buffer.observations[idx]
        true_next_obs, *_ = env.step(online_buffer.actions[idx])
        online_buffer.true_next_observations[idx] = true_next_obs

    # np.savez_compressed(
    #     "rollout.npz",
    #     observations=online_buffer.observations,
    #     actions=online_buffer.actions,
    #     predicted=online_buffer.predicted_next_observations,
    #     true=online_buffer.true_next_observations
    # )

def setup_environment_and_dataset(args):
    env = gym.make(args.task)
    env.reset()
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    return env, dataset

def setup_logging(args):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{timestamp}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer, log_path=log_path)

    set_device_and_logger(0 if args.device == 'cuda' else -1, logger)

    return writer, logger, log_path

def compute_correlation(online_buffer):
    dist = compute_l2_distance(online_buffer.predicted_next_observations, online_buffer.true_next_observations)
    uncertainty = online_buffer.penalties.flatten()
    spearman = compute_spearman_rank(dist, uncertainty)
    pearson = compute_pearson_rank(dist, uncertainty)
    return spearman, pearson, dist, uncertainty

def build_sac_policy(args, env):
    # Actor and Critics
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic_input_dim = np.prod(args.obs_shape) + args.action_dim
    critic1_backbone = MLP(input_dim=critic_input_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=critic_input_dim, hidden_dims=[256, 256])

    dist = DiagGaussian(
        latent_dim=actor_backbone.output_dim,
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy or -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    return SACPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_optim,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

def load_model_components(dynamics_model, policy, model_path, policy_path):
    dynamics_model.model = torch.load(model_path)
    policy.load_state_dict(torch.load(policy_path))

def compare_uncertainty_measures(args=get_args()):
    env, dataset = setup_environment_and_dataset(args)
    # writer, logger, log_path = setup_logging(args)
    set_device_and_logger(0 if args.device == 'cuda' else -1, None)

    # Dynamic configuration
    task_prefix = args.task.split('-')[0]
    static_fns = importlib.import_module(f"static_fns.{task_prefix}").StaticFns
    config = importlib.import_module(f"config.{task_prefix}").default_config

    sac_policy = build_sac_policy(args, env)

    dynamics_model = TransitionModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        static_fns=static_fns,
        lr=args.dynamics_lr,
        uncertainty=args.uncertainty,
        dataset=dataset,
        **config["transition_params"]
    )

    # model_path = os.path.join(args.logdir, args.task, args.algo_name, f'seed_{args.seed}_model.pt')
    # policy_path = os.path.join(args.logdir, args.task, args.algo_name, f'seed_{args.seed}_policy.pth')
    base_path = f'./log/{args.task}/mopo/'
    base_path += [f for f in os.listdir(base_path) if f.startswith(f'seed_{args.seed}')][0]
    model_path = os.path.join(base_path, 'models/ite_dynamics_model/model.pt')
    policy_path = os.path.join(base_path, 'policy.pth')
    load_model_components(dynamics_model, sac_policy, model_path, policy_path)

    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)

    model_buffer = RolloutBuffer(
        buffer_size=ROLLOUT_LENGTH * SAMPLE_SIZE,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )

    rollout_data(
        dynamics_model=dynamics_model,
        policy=sac_policy,
        offline_buffer=offline_buffer,
        online_buffer=model_buffer,
        env=env,
        sample_size=SAMPLE_SIZE,
        rollout_length=ROLLOUT_LENGTH
    )

    spearman, pearson, dist, uncertainty = compute_correlation(model_buffer)
    results = pd.DataFrame({
        'dataset': args.task,
        'seed': args.seed,
        'uncertainty': args.uncertainty,
        'spearman': spearman,
        'pearson': pearson
    }, index=[0])
    results.to_csv(os.path.join('outputs', f'results_{args.uncertainty}_{args.task}_{args.seed}.csv'), index=False)
    plot_uncertainty_vs_distance(dist, uncertainty, save_path=os.path.join('outputs', f'uncertainty_vs_distance_{args.uncertainty}_{args.task}_{args.seed}.png'))
    # np.savez_compressed(os.path.join(log_path, f'dist_uncertainty_{args.uncertainty}_{args.task}_{args.seed}.npz'), dist=dist, uncertainty=uncertainty)



if __name__ == '__main__':
    compare_uncertainty_measures()
