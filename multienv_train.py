import datetime
import os
from argparse import ArgumentParser
import shutil
from distutils.util import strtobool
import numpy as np
import torch
import yaml
from vectorized_sc2_env import VecStarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import wandb
from MultiAgentTransformer import MultiAgentTransformer
from buffer import EpisodeBuffer
from collections import defaultdict
import subprocess

def get_current_branch(repository_dir="./") -> str:
    """
    get current branch name

    Args:
        repository_dir(str): the direcory which a reposiory exists

    Returns:
        branch_name(str)
    """
    cmd = "cd %s && git rev-parse --abbrev-ref HEAD" % repository_dir
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    stdout_data = proc.stdout.read()
    # stderr_data = proc.stderr.read()
    current_branch = stdout_data.decode('utf-8').replace('\n','')
    
    return current_branch

def evaluation(
    env: VecStarCraft2Env,
    model: MultiAgentTransformer,
    n_eval: int,
):
    model.eval()
    device = model.device
    eval_info = defaultdict(list)
    obs, action_mask = env.reset()
    while len(eval_info["battle_won"]) < n_eval:
        with torch.no_grad():
            action, action_logps, entropy, values = model.get_action_and_value(
                torch.tensor(obs, dtype=torch.float32, device=device),
                torch.tensor(action_mask, dtype=torch.int32, device=device),
                deterministic=True,
            )
        (obs, action_mask), rewards, dones, truncateds, infos = env.step(
            action.detach().cpu().numpy()
        )
        # Check the battle results
        for env_id, truncated in enumerate(truncateds):
            if truncated and "battle_won" in infos[env_id]:
                # Finish one episode
                print(f"Eval [{env_id}]: info: {infos[env_id]}")
                for key in [
                    "battle_won",
                    "total_rewards",
                    "episode_length",
                    "dead_allies",
                    "dead_enemies",
                ]:
                    eval_info[key].append(infos[env_id][key])

    return eval_info


def main(args):
    n_training_threads = args.n_training_threads
    save_path = args.save_path
    config_path = args.config_path
    debug = args.debug
    n_eval_episodes = args.n_eval_episodes
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    n_steps = int(float(data["n_steps"]))
    n_ppo_update = int(data["n_ppo_update"])
    time_horizon = int(data["time_horizon"])
    gamma = data["gamma"]
    tau = data["tau"]
    max_batch_size = 10000
    n_env = data["n_env"]

    if not "target_kl" in data:
        # Default target kl for early stopping
        data["target_kl"] = 0.01

    if not "shuffle_agent_idx" in data:
        data["shuffle_agent_idx"] = False

    # Setting up tensorboard
    date = datetime.datetime.now()
    run_name = f"{data['map_name']}-{date.month}-{date.day}-{date.hour}-{date.minute}"
    if args.track:
        config_dict = vars(args)
        config_dict.update(data)
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=config_dict,
            name=run_name + save_path,
            save_code=True,
        )
    # Summary writer
    branch_name = get_current_branch()
    save_dir = "models/" + save_path + f"/{branch_name + run_name}/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if not debug:
        writer = SummaryWriter(save_dir + "logs/")

    torch.set_num_threads(n_training_threads)

    # Setup vectorized envs
    env = VecStarCraft2Env(n_env=n_env, map_name=data["map_name"], remote_envs=True, version=data["version"])
    eval_env = VecStarCraft2Env(n_env=2, map_name=data["map_name"], remote_envs=True, version=data["version"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.obs_dim

    model = MultiAgentTransformer(
        data["n_dim"],
        data["n_head"],
        data["num_layer_encoder"],
        obs_dim,
        env.action_dim,
        data["num_layer_decoder"],
        n_agent=env.n_agents,
        gamma=data["gamma"],
        clip=data["clip"],
        lr=float(data["lr"]),
        eps=float(data["eps"]),
        entropy_coef=data["entropy_coef"],
        max_grad_norm=data["max_grad_norm"],
        huber_delta=data["huber_delta"],
        device=device,
    )

    def show_parameters(model):
        # count the volume of parameters of model
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue
        print(f"Total params: {Total_params}")
        print(f"Trainable params: {Trainable_params}")
        print(f"Non-trainable params: {NonTrainable_params}")

    print("\n==== model ====")
    show_parameters(model)

    print("\n==== model.encoder====")
    show_parameters(model.encoder)

    print("\n==== model.decoder ====")
    show_parameters(model.decoder)

    # Reset environment
    obs, action_mask = env.reset()
    try:
        for update in range(n_steps // (n_env * time_horizon)):
            # Rollout
            model.eval()
            eps_buffer = EpisodeBuffer(
                env.n_env,
                env.n_agents,
                obs_dim,
                env.action_dim,
                gamma,
                tau,
                time_horizon,
                max_batch_size,
                device,
                shuffle_agent_idx=data["shuffle_agent_idx"],
            )
            rollout_info = defaultdict(list)
            while not eps_buffer.is_full():
                with torch.no_grad():
                    action, action_logps, entropy, values = model.get_action_and_value(
                        torch.tensor(obs, dtype=torch.float32, device=device),
                        torch.tensor(action_mask, dtype=torch.int32, device=device),
                        deterministic=False,
                    )

                (next_obs, next_action_mask), reward, done, truncateds, info = env.step(
                    action.detach().cpu().numpy()
                )

                # Check if any environments are done
                for env_id, truncated in enumerate(truncateds):
                    if truncated and "battle_won" in info[env_id]:
                        # Finish one episode
                        total_reward = info[env_id]["total_rewards"]
                        rollout_info["total_rewards"].append(total_reward)
                        print(f"Train [{env_id}]: total rewards: {total_reward}")
                        for key in [
                            "battle_won",
                            "episode_length",
                            "dead_allies",
                            "dead_enemies",
                        ]:
                            rollout_info[key].append(info[env_id][key])

                # Add in buffer
                eps_buffer.insert(
                    torch.tensor(obs, dtype=torch.float32, device=device),
                    action,
                    torch.tensor(action_mask, dtype=torch.int32, device=device),
                    action_logps,
                    torch.tensor(reward, dtype=torch.float32, device=device),
                    torch.tensor(done, dtype=torch.float32, device=device),
                    torch.tensor(truncateds, dtype=torch.float32, device=device),
                    values,
                )
                obs = next_obs
                action_mask = next_action_mask
            # Attention: obs and action_mask are used in the next rollout so don't overwrite it bellow
            # Add next value at the end of time_horizon
            with torch.no_grad():
                next_values = model.get_value(
                    torch.tensor(obs, dtype=torch.float32, device=device)
                )
            eps_buffer.add_next_value(next_values)
            eps_buffer.compute_advantages(model.value_normalizer)

            # Recode info related to rollout
            tag = "rollout"
            global_step = update * n_env * time_horizon
            for k, v in rollout_info.items():
                if not debug:
                    writer.add_scalar(f"{tag}/{k}_mean", np.mean(v), global_step)
                    writer.add_scalar(f"{tag}/{k}_max", np.max(v), global_step)
                    writer.add_scalar(f"{tag}/{k}_min", np.min(v), global_step)
                    writer.add_scalar(f"{tag}/{k}_std", np.std(v), global_step)

            # PPO update
            train_info = {
                "value_loss": 0,
                "policy_loss": 0,
                "entropy": 0,
                "grad_norm": 0,
                "ratio": 0,
                "approx_kl": 0,
                "clip_frac": 0,
                "explained_var": 0,
            }

            for j in range(n_ppo_update):
                batch = eps_buffer.sample(isall=True)
                (
                    critic_loss,
                    policy_loss,
                    grad_norm,
                    entropy,
                    ratio,
                    approx_kl,
                    clip_frac,
                    explained_var,
                ) = model.update(batch)
                train_info["value_loss"] += critic_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["grad_norm"] += grad_norm
                train_info["entropy"] += entropy.item()
                train_info["ratio"] += ratio.mean()
                train_info["approx_kl"] += approx_kl
                train_info["clip_frac"] += clip_frac
                train_info["explained_var"] += explained_var

                if "target_kl" in data and approx_kl > data["target_kl"]:
                    print(f"Early stopping in epoch {j}")
                    print(f"Target KL: {data['target_kl']:.4f}, KL: {approx_kl:.4f}")
                    break

            for k, v in train_info.items():
                v /= n_ppo_update
                if not debug:
                    writer.add_scalar(f"train/{k}", v, update * n_env * time_horizon)

            if update % 20 == 0:
                # Evaluation
                eval_info = evaluation(eval_env, model, n_eval_episodes)
                tag = "eval"
                global_step = update * n_env * time_horizon
                for k, v in eval_info.items():
                    if not debug:
                        writer.add_scalar(f"{tag}/{k}_mean", np.mean(v), global_step)
                        writer.add_scalar(f"{tag}/{k}_max", np.max(v), global_step)
                        writer.add_scalar(f"{tag}/{k}_min", np.min(v), global_step)
                        writer.add_scalar(f"{tag}/{k}_std", np.std(v), global_step)
                        writer.add_histogram(f"eval_hist/{k}", np.array(v), global_step)

            if update % 500 == 0:
                # Save checkpoint
                model.save_model(save_dir + f"{data['map_name']}-episode-{update}")

    finally:
        env.close()
        if args.track:
            wandb.finish()
        if not debug:
            model.save_model(save_dir + data["map_name"])
            with open(save_dir + "train.yaml", "w") as f:
                yaml.safe_dump(data, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_training_threads", type=int, default=16)
    parser.add_argument("--save-path", type=str, default="hoge")
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--n-eval-episodes", type=int, default=200)
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="MAT-yamashita")

    args = parser.parse_args()

    main(args)
