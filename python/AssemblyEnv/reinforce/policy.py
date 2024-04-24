import os
import numpy as np
from AssemblyEnv.reinforce.env import AssemblyPlayground
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
from AssemblyEnv.geometry import Assembly2D
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

class AssemblyNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim: int = 64,
    ):
        super(AssemblyNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_vf = self.latent_dim_pi = last_layer_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim), nn.ReLU(),
            nn.Linear(last_layer_dim, last_layer_dim), nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim), nn.ReLU(),
            nn.Linear(last_layer_dim, last_layer_dim), nn.ReLU(),
        )


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class AssemblyACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(AssemblyACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.ninf = -1E8

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AssemblyNetwork(self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_obs(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent_obs(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes and observation.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions = (obs) * self.ninf + mean_actions
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_obs(latent_pi, obs)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent_obs(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

class RobotACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        n_part: int = 0,
        n_robot: int = 0,
        *args,
        **kwargs,
    ):

        super(RobotACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.ninf = -1E8
        self.n_part = n_part
        self.n_robot = n_robot


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AssemblyNetwork(self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_obs(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent_obs(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes and observation.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        install_state = obs[:,  :self.n_part]
        fixed_state = obs[:, self.n_part:]

        #non_install_action = th.sum(fixed_state, axis = -1) >= self.n_robot
        mean_actions[:, : self.n_part] = (install_state) * self.ninf + mean_actions[:, : self.n_part]
        #mean_actions[:, : self.n_part] += th.einsum("i, j -> ij", non_install_action, self.ninf * th.ones(self.n_part, dtype=mean_actions.dtype, device = mean_actions.device))
        mean_actions[:, self.n_part:] = (1 - fixed_state) * self.ninf + mean_actions[:, self.n_part:]
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_obs(latent_pi, obs)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent_obs(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy