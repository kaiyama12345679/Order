import torch
from argparse import ArgumentParser
from MultiAgentTransformer import MultiAgentTransformer
from vectorized_sc2_env import VecStarCraft2Env
import time
import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config/train.yaml")
    parser.add_argument("--checkpoint", type=str, default="./models/xxxx")
    parser.add_argument("--n-episode", type=int, default=20)
    args = parser.parse_args()
    # Load config
    with open(args.config_path, "r") as f:
        data = yaml.safe_load(f)

    env = VecStarCraft2Env(n_env=1, map_name=data["map_name"], remote_envs=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiAgentTransformer(
        data["n_dim"],
        data["n_head"],
        data["num_layer_encoder"],
        env.obs_dim,
        env.action_dim,
        data["num_layer_decoder"],
        n_agent=env.n_agents,
        gamma=data["gamma"],
        clip=data["clip"],
        lr=float(data["lr"]),
        eps=float(data["eps"]),
        entropy_coef=data["entropy_coef"],
        max_grad_norm=data["max_grad_norm"],
        device=device,
    )

    try:
        model.load_model(args.checkpoint)
        model.eval()
        counter = 0
        obs, action_mask = env.reset()
        while args.n_episode > counter:
            with torch.no_grad():
                action, action_probs, entropy, values = model.get_action_and_value(
                    torch.tensor(obs, dtype=torch.float32, device=device),
                    torch.tensor(action_mask, dtype=torch.int32, device=device),
                    deterministic=True,
                )
            (obs, action_mask), rewards, dones, truncateds, infos = env.step(
                action.detach().cpu().numpy()
            )
            env.render()
            time.sleep(0.01)

            # Check the battle results
            for env_id, truncated in enumerate(truncateds):
                if truncated:
                    # Finish one episode
                    print(f"Eval [{env_id}]: info: {infos[env_id]}")
                    counter += 1
    finally:
        env.close()
