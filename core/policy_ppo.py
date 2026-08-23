import time
from dataclasses import dataclass
from typing import Tuple
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import pdb
from core.nets import  MLPL,  LinearNet

def find_params(SML):
    
    params = None
    
    if SML.strategy == "MLP":
        
        params = list(SML.poly.parameters()) 

             
    return params


class PPO_MLP_strategy(nn.Module):
    def __init__(self, act_dim, obs_dim,input_dim,layers,layers_critic,omniscent):
        super().__init__()
        
        mult=1   
        self.act_dim=act_dim
        self.obs_dim=obs_dim
  
        self.omniscent=omniscent
         
        self.net = MLPL(in_size=input_dim, out_size=act_dim,layers=(mult*(1+input_dim),)*(1+layers))    

        self.mean = nn.Linear(act_dim, act_dim)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic =  MLPL(in_size=obs_dim, out_size=1,layers=(mult*(1+input_dim),)*(1+layers_critic)) 

    def forward(self, obs):
        
        if self.omniscent:
            x = self.net(obs)  
        else:
            x= self.net(obs[...,self.act_dim:]) 
        
        mean = self.mean(x)
 
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value 



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,omniscent, layers, layer_c):
        super().__init__()
        self.omniscent=omniscent
        if omniscent:
            input_dim=obs_dim
        else:
            input_dim=obs_dim-act_dim
   
        self.poly=PPO_MLP_strategy( act_dim, obs_dim,input_dim,layers,layer_c,omniscent)
      
    def forward(self, obs):
            
        mean, std, value=self.poly(obs)
        return mean, std, value

def compute_gae(rewards, values, next_value, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_non_terminal_value = 0.0
            gae = 0.0
        else:
            next_non_terminal_value = next_value

        delta = rewards[t] + gamma * next_non_terminal_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    return advantages, returns

def ppo_update(model, optimizer, scheduler,obs, acts, old_log_probs, advantages, returns, clip_epsilon, batch_size):
    dataset_size = obs.shape[0]
    #indices = torch.randperm(dataset_size)
    
   
   
    mean, std, values = model(obs)
    dist = Normal(mean, std)
    log_probs = dist.log_prob(acts).sum(dim=-1)
  
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = ((values - returns) ** 2).mean()
    loss = policy_loss + 0.5 * value_loss
   
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()