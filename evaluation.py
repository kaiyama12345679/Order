import torch
from argparse import ArgumentParser
from MultiAgentTransformer import MultiAgentTransformer
from harl.utils.envs_tools import *
import time
import yaml
import numpy as np
from collections import defaultdict

def evaluation(env, model: MultiAgentTransformer, n_eval: int):

    obs, share_obs, action_mask = env.reset()
    if action_mask[0] is None:
        action_mask = np.ones((obs.shape[0], obs.shape[1], model.action_dim)).tolist()
    n_env = len(obs)
    total_rewards = np.zeros((n_env))
    eval_info = defaultdict(list)
    while len(eval_info["total_rewards"]) < n_eval:
        with torch.no_grad():
            action, action_logps, entropy, order, order_logps, order_entropy, values = model.get_action_and_value(
                torch.tensor(obs, dtype=torch.float32, device=device),
                torch.tensor(action_mask, dtype=torch.int32, device=device),
                deterministic=True,
            )
        obs, share_obs, rewards, dones, infos, action_mask = env.step(
            action.detach().cpu().numpy()
        )
        if action_mask[0] is None:
            action_mask = np.ones_like(action.detach().cpu().numpy()).tolist()
        total_rewards += np.array(rewards[:, 0, 0])
        # Check the battle results
        for env_id, done in enumerate(dones):
            if done[0]:
                # Finish one episode
                print(f"Eval [{env_id}]: info: {total_rewards[env_id]}")
                eval_info["total_rewards"].append(total_rewards[env_id])
                total_rewards[env_id] = 0

    return eval_info


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config/train.yaml")
    parser.add_argument("--checkpoint", type=str, default="./models/xxxx")
    parser.add_argument("--n-episode", type=int, default=20)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        data = yaml.safe_load(f)

    env = make_eval_env(data["env_name"], data["eval_seed"], data["n_eval_env"], data["env_args"])

    obs_space = env.observation_space
    action_space = env.action_space
    obs_shape = get_shape_from_obs_space(obs_space)
    sample_obs, sample_share_obs, _ = env.reset()
    obs_dim = len(sample_obs[0][0])
    if data["discrete"]:
        action_dim = action_space[0].n
    else:
        action_dim = action_space[0].shape[0]
        action_low = action_space[0].low
        action_high = action_space[0].high
    action_type = action_space.__class__.__name__

    num_agents = get_num_agents(data["env_name"],data["env_args"], env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = dict()
    
    model = MultiAgentTransformer(
        data["n_dim"],
        data["n_head"],
        data["num_layer_encoder"],
        obs_dim,
        action_dim,
        data["num_layer_decoder"],
        n_agent=num_agents,
        gamma=data["gamma"],
        clip=data["clip"],
        lr=float(data["lr"]),
        eps=float(data["eps"]),
        entropy_coef=data["entropy_coef"],
        max_grad_norm=data["max_grad_norm"],
        huber_delta=data["huber_delta"],
        device=device,
        discrete=data["discrete"]
    )
    try:
        model.load_model(args.checkpoint)
        model.eval()
        eval_info =  evaluation(env, model, args.n_episode)
        for k, v in eval_info.items():
            print(f"{k}_mean: {np.mean(v)}")
            print(f"{k}_std: {np.std(v)}")
    finally:
        env.close()


