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


class Actor_Critic_PFN_V2(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor_Critic_PFN_V2, self).__init__()
        self.fc_input1 = nn.Linear(1 + state_size + action_size + 1, 128) ## input [delta, obs, action + 1]
        self.fc_input2 = nn.Linear(action_size + 1 + state_size + 1, 128) ## input [action + 1, obs, reward]
        self.lstm_1 = nn.LSTM(128, 128, 1, batch_first=True) # (input, hidden, layer) for action
        self.lstm_2 = nn.LSTM(128, 128, 1, batch_first=True) # (input, hidden, layer) for value
        self.a = nn.Linear(128,action_size) ## output each action
        self.v = nn.Linear(128,1) ## output value

    def forward(self, x1, x2, h1, c1, h2, c2):
        batch_size = x1.size(0)
        x1 = torch.relu(self.fc_input1(x1))
        x1.view(batch_size, 1, -1)
        # x = [batch, seq_len, input_size]
        # c, h = [num_layers * num_directions, batch, hidden_size]
        rnn_output_1, (h1_out, c1_out) = self.lstm_1(x1, (h1, c1))
        rnn_output_1 = torch.relu(rnn_output_1)
        a_out = self.a(rnn_output_1).view(batch_size, -1)
        action_distribution = F.softmax(a_out, dim=-1) ## NN에서 각 action에 대한 확률을 추정한다.

        x2 = torch.relu(self.fc_input2(x2))
        x2.view(batch_size, 1, -1)
        rnn_output_2, (h2_out, c2_out) = self.lstm_2(x2, (h2, c2))
        rnn_output_2 = torch.relu(rnn_output_2)
        v_out = self.v(rnn_output_2).view(batch_size, -1)

        return action_distribution, v_out, h1_out, c1_out, h2_out, c2_out

def td_target(v_out, rewards, masks):
    gamma = 0.9
    target_y = torch.zeros_like(v_out)
    values = v_out
    next_value = 0
    
    for t in reversed(range(0,len(rewards))):
        target_y[t] = rewards[t] + gamma*next_value*masks[t] ## 특정 t 시점부터 expected next~t에 해당하는 value가 discounted 됨.
        next_value = values.data[t]
    return target_y

def concat_input_1(last_delta, obs, last_action):
    
    obs = torch.Tensor([obs])
    last_delta = torch.Tensor([[last_delta]])
    onehot_last_action = torch.zeros(1,3)
    onehot_last_action[0][last_action] = 1.0
    concat_input = torch.cat((obs,last_delta,onehot_last_action), -1)

    return concat_input.unsqueeze(0)

def concat_input_2(last_action, obs, last_reward):
    
    onehot_last_action = torch.zeros(1,3)
    onehot_last_action[0][last_action] = 1.0
    obs = torch.Tensor([obs])
    last_reward = torch.Tensor([[last_reward]])
    
    concat_input = torch.cat((onehot_last_action,obs,last_reward), -1)

    return concat_input.unsqueeze(0)


def train(PFN, PFN_optimizer, replay_buffer):
    beta_v = 0.05
    beta_e = 0.05

    # data 분배
    replay_buffer = np.array(replay_buffer)
    input_1 = np.vstack(replay_buffer[:, 0])
    input_2 = np.vstack(replay_buffer[:, 1])
    actions = list(replay_buffer[:, 2])
    rewards = list(replay_buffer[:, 3])
    masks = list(replay_buffer[:, 4]) 
    h1_in = np.vstack(replay_buffer[:, 5])
    c1_in = np.vstack(replay_buffer[:, 6])
    h2_in = np.vstack(replay_buffer[:, 7])
    c2_in = np.vstack(replay_buffer[:, 8])

    # tensor.
    input_1 = torch.Tensor(input_1)
    input_2 = torch.Tensor(input_2)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    h1_in = torch.Tensor(h1_in)
    c1_in = torch.Tensor(c1_in)
    h2_in = torch.Tensor(h2_in)
    c2_in = torch.Tensor(c2_in)

    batch_size = input_1.size(0)

    h1_in = torch.Tensor(h1_in).view(1, batch_size, -1)
    c1_in = torch.Tensor(c1_in).view(1, batch_size, -1)
    h2_in = torch.Tensor(h2_in).view(1, batch_size, -1)
    c2_in = torch.Tensor(c2_in).view(1, batch_size, -1)

    # actor_loss
    action_distribution, v_out, h1_out, c1_out, h2_out, c2_out = PFN(input_1, input_2, h1_in, c1_in, h2_in, c2_in)
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

    PFN = Actor_Critic_PFN_V2(state_space, action_space)
    PFN_optimizer = optim.RMSprop(PFN.parameters(), lr=learning_rate)

    batch_size = 5
    replay_buffer = deque(maxlen=batch_size) # on-policy method로 업데이트 후 data buffer를 초기화 해준다.

    step = 0 ## 총 step을 계산하기 위한 step.
    score = 0
    show_score = []

    for epi in range(episode):
        obs = env.reset() # x0
        input_1 = concat_input_1(0,obs,2)
        input_2 = concat_input_2(2,obs,0)
        h1_in = torch.zeros(1, 1, 128)
        c1_in = torch.zeros(1, 1, 128)
        h2_in = torch.zeros(1, 1, 128)
        c2_in = torch.zeros(1, 1, 128)
        for i in range(maximum_steps):
            
            action_distribution, v_out, h1_out, c1_out, h2_out, c2_out = PFN(input_1, input_2, h1_in, c1_in, h2_in, c2_in) 
            action = get_action(action_distribution)
            
            next_obs, reward, done, _ = env.step(action)
            
            mask = 0 if done else 1

            h1_in = h1_out
            c1_in = c1_out
            h2_in = h2_out
            c2_in = c2_out

            replay_buffer.append((input_1, input_2, action, reward, mask, h1_out.detach().numpy(), c1_out.detach().numpy(), h2_out.detach().numpy(), c2_out.detach().numpy())) ## 저장
            
            score += reward ## reward 갱신.
            step += 1
            
            if step % batch_size == 0 and step != 0:
                train(PFN, PFN_optimizer, replay_buffer) # batch마다 train한다.
                replay_buffer = deque(maxlen=batch_size) # on-policy method로 업데이트 후 data buffer를 초기화 해준다.
            delta = td_target(v_out.detach(), torch.Tensor([reward]), torch.Tensor([mask]))
            input_1 = concat_input_1(delta,next_obs,action) ## next
            input_2 = concat_input_2(action,next_obs,reward) ## next

            if done:
                break
        
        
        if epi % print_interval == 0 and epi != 0:
            show_score.append(score/print_interval) ## reward score 저장.
            print('episode: ',epi,' step: ', step, 'score: ', score/print_interval) # log 출력.
            score = 0
            with open('PFN_v2_batch5.p', 'wb') as file:
                pickle.dump(show_score, file)
    env.close()

if __name__ == '__main__':
    main()