import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl.policy_network import PolicyNetwork, ValueNetwork
from utils.cuda_parallel import CudaParallel

class CudaParallelPPOAgent:
    """
    CUDA并行PPO代理实现
    """
    def __init__(self, state_dim, action_dims, device, 
                 lr=0.0003, gamma=0.99, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        """
        初始化CUDA并行PPO代理
        
        参数:
            state_dim: 状态空间维度
            action_dims: 动作空间维度列表 [剪枝动作数, 转换动作数]
            device: 计算设备
            lr: 学习率
            gamma: 折扣因子
            clip_ratio: PPO裁剪比例
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
        """
        # 初始化CUDA并行工具
        self.cuda_parallel = CudaParallel()
        
        # 如果有CUDA设备可用，使用CUDA
        if self.cuda_parallel.is_available:
            device = torch.device("cuda:0")
        
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 创建策略网络和价值网络
        self.policy_net = PolicyNetwork(state_dim, action_dims).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        # 如果有多个GPU可用，并行化网络
        if self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 1:
            self.policy_net = self.cuda_parallel.parallelize_model(self.policy_net)
            self.value_net = self.cuda_parallel.parallelize_model(self.value_net)
            print(f"PPO代理已启用CUDA并行模式，使用 {self.cuda_parallel.num_gpus} 个GPU")
        
        # 创建优化器
        if isinstance(self.policy_net, nn.DataParallel):
            self.policy_optimizer = optim.Adam(self.policy_net.module.parameters(), lr=lr)
            self.value_optimizer = optim.Adam(self.value_net.module.parameters(), lr=lr)
        else:
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
        
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作分布
        with torch.no_grad():
            if isinstance(self.policy_net, nn.DataParallel):
                prune_probs, convert_probs = self.policy_net.module(state_tensor)
            else:
                prune_probs, convert_probs = self.policy_net(state_tensor)
        
        # 从分布中采样动作
        prune_dist = torch.distributions.Categorical(probs=prune_probs)
        prune_action = prune_dist.sample().item()
        
        convert_dist = torch.distributions.Categorical(probs=convert_probs)
        convert_action = convert_dist.sample().item()
        
        # 计算动作的对数概率
        log_prob_prune = prune_dist.log_prob(torch.tensor(prune_action).to(self.device))
        log_prob_convert = convert_dist.log_prob(torch.tensor(convert_action).to(self.device))
        log_prob = log_prob_prune + log_prob_convert
        
        return [prune_action, convert_action], log_prob.item()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        """
        存储经验
        
        参数:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作的对数概率
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update(self, batch_size=64, epochs=10):
        """
        更新策略和价值网络
        
        参数:
            batch_size: 批次大小
            epochs: 更新轮数
        """
        # 如果经验不足，不更新
        if len(self.states) < batch_size:
            return
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # 计算回报
        returns = self.compute_returns(rewards, dones)
        
        # 计算优势
        with torch.no_grad():
            if isinstance(self.value_net, nn.DataParallel):
                values = self.value_net.module(states).squeeze()
                next_values = self.value_net.module(next_states).squeeze()
            else:
                values = self.value_net(states).squeeze()
                next_values = self.value_net(next_states).squeeze()
            
            # 计算TD误差
            td_errors = rewards + self.gamma * next_values * (1 - dones) - values
            
            # 计算广义优势估计(GAE)
            advantages = self.compute_gae(td_errors, dones)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 如果有多个GPU，使用并行批处理
        if self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 1:
            # 扩大批次大小以充分利用多GPU
            batch_size = batch_size * self.cuda_parallel.num_gpus
        
        # 多轮更新
        for _ in range(epochs):
            # 生成批次索引
            indices = np.random.permutation(len(states))
            
            # 按批次更新
            for start_idx in range(0, len(states), batch_size):
                # 获取批次索引
                idx = indices[start_idx:start_idx + batch_size]
                
                # 提取批次数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # 更新策略网络
                self.update_policy(batch_states, batch_actions, batch_old_log_probs, batch_advantages)
                
                # 更新价值网络
                self.update_value(batch_states, batch_returns)
        
        # 更新后清空经验缓冲区
        self.clear_experience()
    
    def clear_experience(self):
        """
        清空经验缓冲区
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def update_policy(self, states, actions, old_log_probs, advantages):
        """
        更新策略网络
        
        参数:
            states: 状态批次
            actions: 动作批次
            old_log_probs: 旧的动作对数概率
            advantages: 优势估计
        """
        # 计算当前策略的动作概率
        if isinstance(self.policy_net, nn.DataParallel):
            prune_probs, convert_probs = self.policy_net.module(states)
        else:
            prune_probs, convert_probs = self.policy_net(states)
        
        # 计算当前动作的对数概率
        prune_actions = actions[:, 0]
        convert_actions = actions[:, 1]
        
        prune_dist = torch.distributions.Categorical(probs=prune_probs)
        convert_dist = torch.distributions.Categorical(probs=convert_probs)
        
        log_prob_prune = prune_dist.log_prob(prune_actions)
        log_prob_convert = convert_dist.log_prob(convert_actions)
        
        new_log_probs = log_prob_prune + log_prob_convert
        
        # 计算重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算策略损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算熵损失（用于探索）
        entropy = prune_dist.entropy().mean() + convert_dist.entropy().mean()
        
        # 总损失
        loss = policy_loss - self.entropy_coef * entropy
        
        # 优化
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
    
    def update_value(self, states, returns):
        """
        更新价值网络
        
        参数:
            states: 状态批次
            returns: 回报批次
        """
        # 计算当前状态的价值估计
        if isinstance(self.value_net, nn.DataParallel):
            values = self.value_net.module(states).squeeze()
        else:
            values = self.value_net(states).squeeze()
        
        # 计算价值损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 优化
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def compute_returns(self, rewards, dones, gamma=None):
        """
        计算折扣回报
        
        参数:
            rewards: 奖励序列
            dones: 结束标志
            gamma: 折扣因子，如果为None则使用self.gamma
        
        返回:
            returns: 折扣回报
        """
        if gamma is None:
            gamma = self.gamma
        
        returns = []
        R = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + gamma * R * (1 - done)
            returns.insert(0, R)
        
        return torch.tensor(returns).to(self.device)
    
    def compute_gae(self, td_errors, dones, gamma=None, lambda_=0.95):
        """
        计算广义优势估计(GAE)
        
        参数:
            td_errors: TD误差
            dones: 结束标志
            gamma: 折扣因子，如果为None则使用self.gamma
            lambda_: GAE参数
        
        返回:
            advantages: 优势估计
        """
        if gamma is None:
            gamma = self.gamma
        
        advantages = []
        gae = 0
        
        for td_error, done in zip(reversed(td_errors), reversed(dones)):
            gae = td_error + gamma * lambda_ * gae * (1 - done)
            advantages.insert(0, gae)
        
        return torch.tensor(advantages).to(self.device)
    
    def save(self, path):
        """
        保存模型参数
        
        参数:
            path: 保存路径
        """
        if isinstance(self.policy_net, nn.DataParallel):
            torch.save({
                'policy_net': self.policy_net.module.state_dict(),
                'value_net': self.value_net.module.state_dict()
            }, path)
        else:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'value_net': self.value_net.state_dict()
            }, path)
    
    def load(self, path):
        """
        加载模型参数
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(self.policy_net, nn.DataParallel):
            self.policy_net.module.load_state_dict(checkpoint['policy_net'])
            self.value_net.module.load_state_dict(checkpoint['value_net'])
        else:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net']) 