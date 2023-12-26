import torch
from argparse import ArgumentParser
from MultiAgentTransformer import MultiAgentTransformer
from harl.utils.envs_tools import *
import time
import yaml
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config/train.yaml")
    parser.add_argument("--checkpoint", type=str, default="./models/xxxx")
    parser.add_argument("--n-episode", type=int, default=20)
    args = parser.parse_args()
    # Load config
    with open(args.config_path, "r") as f:
        data = yaml.safe_load(f)

    env, manual_render, manual_expand_dims, manual_delay, env_num = make_render_env(data["env_name"], data["eval_seed"], data["env_args"])
    obs_space = env.observation_space
    action_space = env.action_space
    obs_shape = get_shape_from_obs_space(obs_space)
    sample_obs, sample_share_obs, _ = env.reset()
    if manual_expand_dims:
        obs_dim = len(sample_obs[0])
    else:
        obs_dim = len(sample_obs[0][0])
    if data["discrete"]:
        action_dim = action_space[0].n
    else:
        action_dim = action_space[0].shape[0]
        action_low = action_space[0].low
        action_high = action_space[0].high
    action_type = action_space.__class__.__name__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_agents = get_num_agents(data["env_name"],data["env_args"], env)

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
        counter = 0
        cnt = 0
        obs, share_obs, _ = env.reset()
        total_rewards = np.zeros((env_num))
        if manual_expand_dims:
            obs = np.array(obs)[np.newaxis, :, :]
        if True:
            action_mask = np.ones((obs.shape[0], obs.shape[1], model.action_dim)).tolist()
        while 1000 > counter:
            with torch.no_grad():
                action, action_logps, entropy, order, order_logps, order_entropy, values = model.get_action_and_value(
                    torch.tensor(obs, dtype=torch.float32, device=device),
                    torch.tensor(action_mask, dtype=torch.int32, device=device),
                    deterministic=True,
                )

            if manual_expand_dims:
                action = action.squeeze(0)
            obs, share_obs, rewards, dones, infos, _ = env.step(
                action.detach().cpu().numpy()
            )
            cnt += 1
            if manual_expand_dims:
                obs = np.array(obs)[np.newaxis, :, :]
            print(order)
            if manual_render:
                env.render()
            if manual_delay:
                time.sleep(0.02)
            if action_mask[0] is None:
                action_mask = np.ones_like(action.detach().cpu().numpy()).tolist()
            # Check the battle results
            if cnt >= 170:
                cnt = 0
                counter += 1
                obs, share_obs, _ = env.reset()
                if manual_expand_dims:
                    obs = np.array(obs)[np.newaxis, :, :]
                    
    finally:
        env.close()
