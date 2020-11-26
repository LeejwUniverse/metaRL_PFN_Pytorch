import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import time
from collections import deque
import random
import math
import pickle
"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU


class Actor_Critic_PFN(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor_Critic_PFN, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size + 1 + 1, 128) ## input state
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True) # (input, hidden, layer)
        self.a = nn.Linear(128,action_size) ## output each action
        self.v = nn.Linear(128,1)

    def forward(self, x, h, c):
        batch_size = x.size(0)
        x = torch.relu(self.fc1(x))
        x.view(batch_size, 1, -1)
        # x = [batch, seq_len, input_size]
        # c, h = [num_layers * num_directions, batch, hidden_size]
        rnn_output, (h_out, c_out) = self.lstm(x, (h, c))
        rnn_output = torch.relu(rnn_output)
        a_out = self.a(rnn_output).view(batch_size, -1)

        action_distribution = F.softmax(a_out, dim=-1) ## NN에서 각 action에 대한 확률을 추정한다.

        v_out = self.v(rnn_output).view(batch_size, -1)

        return action_distribution, v_out, h_out, c_out

def get_action(action_distribution):

    distribution = Categorical(action_distribution) ## pytorch categorical 함수는 array에 담겨진 값들을 확률로 정해줍니다.
    action = distribution.sample().item()
    
    return action

def td_target(v_out, rewards, masks):
    gamma = 0.9
    
    target_y = torch.zeros_like(v_out)
    values = v_out
    next_value = 0
    
    for t in reversed(range(0,len(rewards))):
        target_y[t] = rewards[t] + gamma*next_value*masks[t] ## 특정 t 시점부터 expected next~t에 해당하는 value가 discounted 됨.
        next_value = values.data[t]
    return target_y

def concat_input(obs, last_reward, last_action):
    
    obs = torch.Tensor([obs])
    reward = torch.Tensor([[last_reward]])
    onehot_last_action = torch.zeros(1,3)
    onehot_last_action[0][last_action] = 1.0
    concat_input = torch.cat((obs,reward,onehot_last_action), -1)

    return concat_input.unsqueeze(0)

def train(PFN, PFN_optimizer, replay_buffer):
    beta_v = 0.05
    beta_e = 0.05

    # data 분배
    replay_buffer = np.array(replay_buffer)
    input_total = np.vstack(replay_buffer[:, 0]) 
    actions = list(replay_buffer[:, 1])
    rewards = list(replay_buffer[:, 2])
    next_input_total = np.vstack(replay_buffer[:, 3])
    masks = list(replay_buffer[:, 4]) 
    h_in = np.vstack(replay_buffer[:, 5])
    c_in = np.vstack(replay_buffer[:, 6])
  

    # tensor.
    input_total = torch.Tensor(input_total)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.Tensor(rewards) 
    next_input_total = torch.Tensor(next_input_total)
    masks = torch.Tensor(masks)

    batch_size = input_total.size(0)

    h_in = torch.Tensor(h_in).view(1, batch_size, -1)
    c_in = torch.Tensor(c_in).view(1, batch_size, -1)
    
    # actor_loss
    action_distribution, v_out, h_out, c_out = PFN(input_total, h_in, c_in) ## state, last_action, last_reward
    distribution = Categorical(action_distribution) ## Categorical 함수를 이용해 하나의 분포도로 만들어줍니다.
    entropies = distribution.entropy().mean()
    
    policy = action_distribution.gather(1,actions)
    log_policy = torch.log(policy)

    target_y = td_target(v_out, rewards, masks) ## 시간차 타겟을 구한다. bootstrap
    
    advantages = target_y - v_out ## r_t + gamma * v(s_t+1) - v(s_t) 
    actor_loss = torch.sum(-log_policy*advantages.detach())
    
    # critic_loss
    mse_loss = torch.nn.MSELoss()
    critic_loss = mse_loss(target_y.detach(), v_out)
    
    # total_loss
    total_loss = actor_loss + beta_v * critic_loss + beta_e * entropies

    # backward
    PFN_optimizer.zero_grad()
    total_loss.backward()
    PFN_optimizer.step()

def main():
    
    episode = 100000000
    maximum_steps = 300
    print_interval = 100 ## 몇 episode 마다 log 출력할건지.
   
    learning_rate = 0.0007
    
    env = gym.make('CartPole-v1')

    action_space = 2
    state_space = 4

    PFN = Actor_Critic_PFN(state_space, action_space) 
    PFN_optimizer = optim.RMSprop(PFN.parameters(), lr=learning_rate) 
    
    batch_size = 32
    replay_buffer = deque(maxlen=batch_size) # on-policy method로 업데이트 후 data buffer를 초기화 해준다.

    step = 0 ## 총 step을 계산하기 위한 step.
    score = 0
    show_score = []

    for epi in range(episode):
        obs = env.reset() # x0
        input_total = concat_input(obs,0,2)
        h_in = torch.zeros(1, 1, 128)
        c_in = torch.zeros(1, 1, 128)
        for i in range(maximum_steps):
            
            action_distribution, v_out, h_out, c_out = PFN(input_total, h_in, c_in) ## state, last_action, last_reward
            action = get_action(action_distribution)
            
            next_obs, reward, done, _ = env.step(action)
            
            mask = 0 if done else 1
            
            next_input_total = concat_input(next_obs, reward, action)
            
            replay_buffer.append((input_total, action, reward, next_input_total, mask, h_out.detach().numpy(), c_out.detach().numpy())) ## 저장
            

            input_total = next_input_total ## current state를 이제 next_state로 변경
            h_in = h_out
            c_in = c_out
            score += reward ## reward 갱신.
            step += 1
            
            if step % batch_size == 0 and step != 0:
                train(PFN, PFN_optimizer, replay_buffer) # batch마다 train한다.
                replay_buffer = deque(maxlen=batch_size) # on-policy method로 업데이트 후 data buffer를 초기화 해준다.

            if done:
                break
        
        
        if epi % print_interval == 0 and epi != 0:
            show_score.append(score/print_interval) ## reward score 저장.
            print('episode: ',epi,' step: ', step, 'score: ', score/print_interval) # log 출력.
            score = 0
            with open('PFN_v1_batch32.p', 'wb') as file:
                pickle.dump(show_score, file)
    env.close()

if __name__ == '__main__':
    main()