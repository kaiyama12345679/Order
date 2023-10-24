# https://github.com/oxwhirl/smac/blob/master/smac/examples/rllib/env.py
# With modification and only support ray newer than 2.3.0
from __future__ import annotations

import random
import math
import gymnasium as gym
from collections import defaultdict
from ray import rllib
try:
    from smac.env import StarCraft2Env as V1Env
except:
    pass
try:
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
except:
    pass
from typing import Any, Optional
import ray
from ray.rllib.env.remote_base_env import RemoteBaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper
import numpy as np
import logging


class VecStarCraft2Env:
    def __init__(
        self,
        n_env: int,
        map_name: str,
        remote_envs: bool = True,
        version: int = 1,
        n_agents: int = 10,
        n_enemies: int = 10,
    ) -> None:
        """Initialize the VecStarCraft2 environment.

        Args:
            n_env: Number of environments.
            map_name: Name of the map.
            remote_envs: If the environments are remote.
        """

        def make_env(*args: Any, **kwargs: Any) -> "RLlibStarCraft2Env":
            """Helper function to create a new environment."""
            return RLlibStarCraft2Env(version, n_agents, n_enemies, map_name=map_name, obs_last_action=True)

        self._single_env = make_env()
        self.n_env = n_env
        self.n_agents = len(self._single_env.get_agent_ids())
        self.remote_envs = remote_envs
        self.obs_dim = self._single_env.observation_space["obs"].shape[0]
        self.action_dim = self._single_env.action_space.n

        # Keep previous observations
        self.prev_obs: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))

        # Set remote_env_batch_wait_ms to a high value so that you can get all environment outputs together
        self._env = self._single_env.to_base_env(
            num_envs=n_env,
            make_env=make_env,
            remote_envs=self.remote_envs,
            remote_env_batch_wait_ms=int(10e9),
        )
        self._is_initialized = False
        self.logger = logging.getLogger("VecStarCraft2Env")
        self.logging_level = logging.INFO
        self.logger.setLevel(self.logging_level)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Resets the environments and returns the observations."""
        if not self._is_initialized:
            self._is_initialized = True
            observations, _, _, _, _, _ = self._env.poll()
        else:
            reset_ids = list(range(self.n_env))
            observations = self._try_reset(reset_ids, {})

        # Unwrap everything for easy handling dict -> list
        obs, action_mask = self._separate_obs_action_mask(observations)
        unwrap_obs = self._unwrap(obs)
        unwrap_action_mask = self._unwrap(action_mask)

        # Keep previous observations
        self.prev_obs = (unwrap_obs, unwrap_action_mask)
        return unwrap_obs, unwrap_action_mask

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]
    ]:
        """Execute actions in the environments and get observations, rewards, dones, and infos.

        Args:
            actions: Numpy array of actions for each environment.

        Returns:
            observations, rewards, dones, truncated, infos
        """
        wrap_action = self._wrap(actions)
        self._env.send_actions(wrap_action)
        observations, rewards, dones, truncateds, infos, _ = self._env.poll()

        # Auto reset and update observation since the observation should be after reset
        reset_ids = [
            env_id for env_id, truncated in truncateds.items() if truncated["__all__"]
        ]
        observations = self._try_reset(reset_ids, observations)

        # Unwrap everything for easy handling dict -> list
        obs, action_mask = self._separate_obs_action_mask(observations)
        unwrap_obs = self._unwrap(obs)
        unwrap_action_mask = self._unwrap(action_mask)
        unwrap_rewards = self._unwrap(rewards)
        unwrap_dones = self._unwrap(dones)
        # Done flag in SC2
        unwrap_truncateds = np.array(
            [truncateds[i]["__all__"] for i in range(self.n_env)]
        )[:, np.newaxis, np.newaxis]
        # For info, all agent info should be the same
        unwrap_infos = [info[0] for info in infos.values()]

        # Keep previous observations
        self.prev_obs = (unwrap_obs, unwrap_action_mask)
        return (
            (unwrap_obs, unwrap_action_mask),
            unwrap_rewards,
            unwrap_dones,
            unwrap_truncateds,
            unwrap_infos,
        )

    def _try_reset(
        self, env_ids: list[int], observations: dict[int | str, dict]
    ) -> dict[int | str, dict]:
        """Attempt to reset specified environments.

        Args:
            env_ids: List of environment ids to reset.
            observations: Current observations dictionary.

        Returns:
            Updated observations dictionary.
        """
        if len(env_ids) == 0:
            return observations
        self.logger.debug(f"Try resetting: {env_ids}")
        for env_id in env_ids:
            self._env.try_reset(env_id)
        obs, _, _, _, _, _ = self._env.poll()
        observations.update(obs)
        # if isinstance(self._env, RemoteBaseEnv):
        #     for env_id in env_ids:
        #         self._env.try_reset(env_id)
        #     obs, _, _, _, _, _ = self._env.poll()
        #     observations.update(obs)
        # elif isinstance(self._env, MultiAgentEnvWrapper):
        #     for env_id in env_ids:
        #         obs, _ = self._env.try_reset(env_id)
        #         observations.update(obs)
        return observations

    def render(self, mode: str = "human") -> None:
        """Render the environment.

        Args:
            mode: Rendering mode, defaults to "human".
        """
        if self.remote_envs:
            self.logger.info("No render if remote_envs == True")
            return
        self._env.try_render()

    def close(self) -> None:
        """Close the environment."""
        self._env.stop()

    def __len__(self) -> int:
        """Returns the number of environments."""
        return self.n_env

    def _separate_obs_action_mask(self, obs_action_mask):
        obs = {}
        action_mask = {}
        for env_id in range(self.n_env):
            obs[env_id] = {
                key: val["obs"] for key, val in obs_action_mask[env_id].items()
            }
            action_mask[env_id] = {
                key: val["action_mask"] for key, val in obs_action_mask[env_id].items()
            }
        return obs, action_mask

    def _unwrap(self, dict_values: dict[int | str, dict]) -> np.ndarray:
        single_value = dict_values[0][0]
        if isinstance(single_value, np.ndarray):
            ndim = single_value.shape[0]
            dtype = single_value.dtype
        else:
            ndim = 1
            dtype = type(single_value)
        unwrap_values = np.zeros((self.n_env, self.n_agents, ndim), dtype=dtype)
        for env_id in range(self.n_env):
            for agent_id in range(self.n_agents):
                unwrap_values[env_id, agent_id] = dict_values[env_id][agent_id]
        return unwrap_values

    def _wrap(self, list_values: np.ndarray) -> dict[int | str, dict]:
        wrap_values = defaultdict(dict)
        for env_id in range(self.n_env):
            for agent_id in range(self.n_agents):
                wrap_values[env_id][agent_id] = list_values[env_id][agent_id]
        return dict(wrap_values)


class RLlibStarCraft2Env(rllib.MultiAgentEnv):
    """Wraps a smac StarCraft env to be compatible with RLlib multi-agent."""

    def __init__(self, version, n_agents, n_enemies, **smac_args):
        """Create a new multi-agent StarCraft env compatible with RLlib.

        Arguments:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.
        """
        super().__init__()
        if version == 1:
            self._env = V1Env(**smac_args)
        else:
            distribution_config = {
                "10gen_terran": {
                    "n_units": n_agents,
                    "n_enemies": n_enemies,
                    "team_gen": {
                        "dist_type": "weighted_teams",
                        "unit_types": ["marine", "marauder", "medivac"],
                        "weights": [0.45, 0.45, 0.1],
                        "exception_unit_types": ["medivac"],
                        "observe": True,
                    },
                    "start_positions": {
                        "dist_type": "surrounded_and_reflect",
                        "p": 0.5,
                        "n_enemies": n_enemies,
                        "map_x": 32,
                        "map_y": 32,
                    },
                },
                "10gen_protoss": {
                    "n_units": n_agents,
                    "n_enemies": n_enemies,
                    "team_gen": {
                        "dist_type": "weighted_teams",
                        "unit_types": ["stalker", "zealot", "colossus"],
                        "weights": [0.45, 0.45, 0.1],
                        "observe": True,
                    },
                    "start_positions": {
                        "dist_type": "surrounded_and_reflect",
                        "p": 0.5,
                        "n_enemies": n_enemies,
                        "map_x": 32,
                        "map_y": 32,
                    },
                },
                "10gen_zerg": {
                    "n_units": n_agents,
                    "n_enemies": n_enemies,
                    "team_gen": {
                        "dist_type": "weighted_teams",
                        "unit_types": ["zergling", "hydralisk", "baneling"],
                        "weights": [0.45, 0.45, 0.1],
                        "exception_unit_types": ["baneling"],
                        "observe": True,
                    },
                    "start_positions": {
                        "dist_type": "surrounded_and_reflect",
                        "p": 0.5,
                        "n_enemies": n_enemies,
                        "map_x": 32,
                        "map_y": 32,
                    },
                }
            }
            self._env = StarCraftCapabilityEnvWrapper(
                capability_config=distribution_config[smac_args["map_name"]], 
                conic_fov=False,
                obs_own_pos=True,
                use_unit_ranges=True,
                min_attack_range=2,
                **smac_args
            )
        self._agent_ids = set(range(self._env.n_agents))
        self._last_own_action = True
        self._obs_own_id = True
        self._action_dim = self._env.get_total_actions()
        self._obs_dim = self._env.get_obs_size()
        if self._last_own_action:
            self._obs_dim += self._action_dim
            self._last_action = [0 for _ in range(self._env.n_agents)]
        if self._obs_own_id:
            self._obs_dim += self._env.n_agents

        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(-1, 1, shape=(self._obs_dim,)),
                "action_mask": gym.spaces.Box(0, 1, shape=(self._action_dim,)),
            }
        )
        self.action_space = gym.spaces.Discrete(self._env.get_total_actions())
        self.action_masks = {}
        self._total_rewards = []

        # divided reward should be True forGroupAgentsWrapper otherwise False
        self.divided_reward = False
        # # Auto reset is desirable for vectorized environment
        # self.auto_reset = True

    def _one_hot(self, target_id, max_id):
        eye = np.eye(max_id, dtype=np.float32)
        return eye[target_id]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
            info (dict): Empty dict for compatibility
        """
        obs_list, state_list = self._env.reset()
        self._last_action = [0 for _ in range(self._env.n_agents)]
        return_obs = {}
        for i, obs in enumerate(obs_list):
            self.action_masks[i] = np.array(self._env.get_avail_agent_actions(i))
            # Include last own action as observation
            if self._last_own_action:
                one_hot_action = self._one_hot(self._last_action[i], self._action_dim)
                obs = np.concatenate([obs, one_hot_action], axis=-1)
            # Include own id as observation
            if self._obs_own_id:
                one_hot_id = self._one_hot(i, self._env.n_agents)
                obs = np.concatenate([obs, one_hot_id], axis=-1)

            return_obs[i] = {
                "action_mask": self.action_masks[i],
                "obs": obs,
            }
        self._total_rewards = []
        return return_obs, {}

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            truncated (dict): Dummy dictionary for compatibility
            infos (dict): Optional info values for each agent id.
        """

        actions = []
        for i in self.get_agent_ids():
            if i not in action_dict:
                raise ValueError("You must supply an action for agent: {}".format(i))
            actions.append(action_dict[i])
        self._last_action = np.array(actions).flatten().tolist()

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        dones = {}
        for i, obs in enumerate(obs_list):
            # If the agent die, the observation is all zero
            dones[i] = not np.any(obs)

            self.action_masks[i] = np.array(self._env.get_avail_agent_actions(i))
            # Include last own action as observation
            if self._last_own_action:
                one_hot_action = self._one_hot(self._last_action[i], self._action_dim)
                if dones[i]:
                    # If the agent is done, don't include action as in MAT paper
                    one_hot_action = np.zeros_like(one_hot_action)
                obs = np.concatenate([obs, one_hot_action], axis=-1)
            # Include own id as observation
            if self._obs_own_id:
                one_hot_id = self._one_hot(i, self._env.n_agents)
                obs = np.concatenate([obs, one_hot_id], axis=-1)
            return_obs[i] = {
                "action_mask": self.action_masks[i],
                "obs": obs,
            }
        # cumulative reward
        self._total_rewards.append(rew)
        if self.divided_reward:
            rews = {i: rew / len(obs_list) for i in self.get_agent_ids()}
        else:
            rews = {i: rew for i in self.get_agent_ids()}
        # __all__ should be terminated flag
        dones["__all__"] = done
        if dones["__all__"]:
            info["total_rewards"] = sum(self._total_rewards)
            info["episode_length"] = len(self._total_rewards)
        infos = {i: info for i in self.get_agent_ids()}
        truncated = {"__all__": done}
        # # For vectorize environment, auto reset is desirable
        # if self.auto_reset and dones["__all__"]:
        #     return_obs = self.reset()
        return return_obs, rews, dones, truncated, infos

    def close(self):
        """Close the environment"""
        self._env.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def render(self):
        self._env.render()

    def action_space_sample(self, agent_ids: list = None) -> dict:
        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        actions = {}
        for agent_id in agent_ids:
            action_ids = np.nonzero(self.action_masks[agent_id])[0]
            action = np.random.choice(action_ids)
            actions[agent_id] = action
        return actions


class DummyAgent(object):
    def generate_action(self, observations: tuple[np.ndarray, np.ndarray]):
        obs, action_masks = observations
        actions = []
        for env_id in range(len(action_masks)):
            actions.append(self._generate_action_single_env(action_masks[env_id]))
        return actions

    def _generate_action_single_env(self, action_masks: np.ndarray):
        actions = []
        for agent_id, action_mask in enumerate(action_masks):
            action_ids = np.nonzero(action_mask)[0]
            action = np.random.choice(action_ids)
            actions.append(action)
        return actions


if __name__ == "__main__":
    import time

    map_id = "8m"
    remote_envs = False
    # Render is only possible for remote_envs==False
    if remote_envs:
        render = False
    else:
        render = True
    env = VecStarCraft2Env(n_env=2, map_name=map_id, remote_envs=remote_envs)
    agent = DummyAgent()
    observations = env.reset()

    total_episodes = 10
    counter = 0
    summary_rewards = []
    while total_episodes > counter:
        actions = agent.generate_action(observations)
        observations, rewards, dones, truncateds, infos = env.step(actions)
        print("---reward-done-info---")
        print(rewards.squeeze())
        print(dones.squeeze())
        print(infos)

        if render:
            env.render()
            time.sleep(0.1)

        for env_id, truncated in enumerate(truncateds):
            if truncated:
                # Finish one episode
                print(
                    f"Done [{env_id}]: total rewards: {infos[env_id]['total_rewards']}"
                )
                summary_rewards.append(infos[env_id]["total_rewards"])
                counter += 1

    print("Summary total rewards:", summary_rewards)
    print("Mean rewards:", np.array(summary_rewards).mean())
    env.close()
