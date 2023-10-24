import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value
import torch.nn as nn
import torch
from collections import namedtuple
from buffer import EpisodeBuffer
from argparse import ArgumentParser
from MultiAgentTransformer import MultiAgentTransformer
from torch.utils.tensorboard import SummaryWriter
from smac.env import StarCraft2Env

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=1000000)
    parser.add_argument("--logdir", type=str, default="./logs")
    args = parser.parse_args()
    n_steps = args.n_steps
    logdir = args.logdir
    writer = SummaryWriter(logdir)
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    obs_dim = env.get_obs_size()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiAgentTransformer(256, 1, 6, obs_dim, n_actions, 6, n_agent=n_agents,device=device)
    model.sync_weight()
    epsbuffer = EpisodeBuffer(0.99, 0.95, 100)
    try:
        for i in range(n_steps):
            _ = env.reset()
            done = False
            total_rewards = []
            total_reward = 0
            step = 0
            while not done:
                with torch.no_grad():
                    obs = env.get_obs()
                    state = env.get_state()
                    action_mask = []
                    for i in range(n_agents):
                        avail_actions = env.get_avail_agent_actions(i)
                        action_mask.append(avail_actions)

                    action, action_probs, values = model.generate_actions(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0), 
                                                                          torch.tensor(action_mask, dtype=torch.int32, device=device).unsqueeze(0))
                    
                    action = action.detach().squeeze(0).cpu().numpy()
                    action_probs = action_probs.detach().squeeze(0).cpu().numpy()
                    values = values.detach().squeeze().cpu().numpy()
                    reward, done, _ = env.step(action)
                    next_obs = env.get_obs()
                    total_reward += reward
                    epsbuffer.insert(obs, action, action_mask, action_probs, reward, next_obs, done, values)

                    step += 1
                    if step >= 100:
                        step = 0
                        next_values = model.calc_values(torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))
                        epsbuffer.add_next_value(next_values.detach().squeeze().cpu().numpy())
                        epsbuffer.compute_advantages()
                        model.buffer.add_steps(epsbuffer)
                        epsbuffer.clear()

            next_values = model.calc_values(torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))
            epsbuffer.add_next_value(next_values.detach().squeeze().cpu().numpy())
            epsbuffer.compute_advantages()
            model.buffer.add_steps(epsbuffer)
            epsbuffer.clear()   
            total_rewards.append(total_reward)
            print("epoch: {}, total_reward: {}".format(i, total_reward))
            if len(model.buffer) > 3200:
                print("training")
                for j in range(50):                
                    model.train()
                    if j % 5 == 0:
                        model.sync_weight()       
                model.clear_buffer()
            writer.add_scalar('reward', total_reward, i)
        model.save_model("model")
    except KeyboardInterrupt as e:
        model.save_model("model")
    finally:
        env.close()
