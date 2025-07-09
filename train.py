import argparse
import datetime
import os
import random
import importlib

import gym
import d4rl

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from models.ae_model import AutoencoderModel
from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger, Scaler, get_args


def rollout(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
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
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     uncertainty=args.uncertainty,
                                     dataset=dataset,
                                     **config["transition_params"]
                                     )
    model_path = ''
    policy_path = ''
    dynamics_model.model = torch.load(model_path)
    #dynamics_model.model.to(args.device)
    sac_policy.load_state_dict(torch.load(policy_path))
    
    obs_list = []
    action_list = []
    next_obs_list = []
    horizon = 100
    episode = 10
    error_list = []
    for i in range(episode):
        obs = env.reset()
        for j in range(horizon):
            action = sac_policy.sample_action(obs, deterministic=True)
            next_obs, penalized_rewards, terminals, info = dynamics_model.predict(obs,action)
            obs_list.append(obs.squeeze())
            action_list.append(action.squeeze())
            next_obs_list.append(next_obs.squeeze())
            
            env.state = obs
            real_next_observations,_,_,_=env.step(action.squeeze())
            obs = next_obs
        #true_next_obs, _*=env.step(action)
        #true_obs_list.append(true_next_obs)
            from sklearn import metrics
            error_list.append(metrics.mean_absolute_error(real_next_observations,next_obs.squeeze()))

    np.savez_compressed(
            "rollout.npz",
            observations=np.array(obs_list),
            actions=np.array(action_list),
            next_observations=np.array(next_obs_list),
        )
    print(error_list)

def train(args=get_args()):
    wandb.init(project="sumo", sync_tensorboard=True)
    # create env and dataset
    env = gym.make(args.task)

    dataset = d4rl.qlearning_dataset(env)

    Scaler.r_max = abs(np.max(dataset['rewards']))
    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
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
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     uncertainty=args.uncertainty,
                                     dataset=dataset,
                                     **config["transition_params"]
                                     )
    autoencoder_model = AutoencoderModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=1e-4,
        dataset=dataset,
    )
    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )

    # create MOPO algo
    algo = MOPO(
        sac_policy,
        dynamics_model,
        autoencoder_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )

    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        eval_episodes=args.eval_episodes
    )

    # # pretrain dynamics model on the whole dataset
    # trainer.train_dynamics()
    #
    # # begin train
    # trainer.train_policy()

    trainer.train_autoencoder()


if __name__ == "__main__":
    train()
    # rollout()
