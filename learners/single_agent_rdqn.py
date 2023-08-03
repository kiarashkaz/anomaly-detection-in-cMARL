from modules.agents.rnn_agent import RNNAgent
from utils.buffer_recurrent import get_device, RecurrentReplayBuffer
import torch
import os
import numpy as np
import math


class RDQNAgent:
    def __init__(self, obs_size, action_size, args, lambda_init=None):
        if lambda_init is None:
            lambda_init = [1/16, 1/16, 1/16, 1/16]
        self.obs_size = obs_size
        self.n_actions = action_size
        self.args = args

        self.gamma = self.args.gamma
        self.lr = self.args.lr
        self.exploration_proba = 1.0
        self.exploration_proba_decay = self.args.exploration_proba_decay
        self.batch_size = self.args.batch_size

        self.buffer = RecurrentReplayBuffer(self.obs_size, 1, self.args.max_episode_length,
                                            self.args.buffer_size, self.args.batch_size)

        self.input_shape = self.obs_size
        self.args.use_rnn = True
        self.model = RNNAgent(self.input_shape, self.args).to(get_device())
        self.hidden = self.model.init_hidden()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.test_mode = False

        self.lambda_coef = np.array(lambda_init)
        self.lambda_lr = 0.02
        self.constraint_value = []

    def load_model(self, save_dir):
        save_name = "adv_model.pth"
        self.model.load_state_dict(
            torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device())))
        self.test_mode = True

    def save_model(self, save_dir):
        save_name = "adv_model.pth"
        torch.save(self.model.state_dict(), os.path.join(save_dir, save_name))

    def compute_action(self, obs, avail_actions):
        avail_actions_ind = np.nonzero(avail_actions)[0]
        if not self.test_mode:
            if np.random.uniform(0, 1) < self.exploration_proba:
                return np.random.choice(avail_actions_ind)
        with torch.no_grad():
            obs = torch.tensor(obs)
            inputs = []
            inputs.append(obs)
            inp = torch.cat([x.reshape(1, -1) for x in inputs], dim=1)
            q_values, self.hidden = self.model(inp, self.hidden)

            for ind in range(self.n_actions):
                if avail_actions[ind] == 0:
                    q_values[0][ind] = -math.inf
            m, i = torch.max(q_values, dim=1)
            return torch.squeeze(i).tolist()

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def train(self):
        batch = self.buffer.sample()
        bs, num_bptt = batch.r.shape[0], batch.r.shape[1]
        obs = torch.FloatTensor(batch.o[:, :, :])
        hidden = self.model.init_hidden().unsqueeze(0).expand(bs, 1, -1)
        predictions = []
        for t in range(num_bptt):
            inp = []
            inp.append(obs[:, t, :])
            inp = torch.cat([x.reshape(bs, -1) for x in inp], dim=1)
            agent_outs, hidden = self.model(inp, hidden)
            assert agent_outs.shape == (bs, self.args.n_actions)
            predictions.append(agent_outs)
        predictions = torch.stack(predictions, dim=1)
        assert predictions.shape == (bs, num_bptt, self.args.n_actions)
        q_values = torch.gather(predictions, dim=2, index=batch.a.to(torch.int64))
        assert q_values.shape == (bs, num_bptt, 1)

        target_hidden = self.model.init_hidden().unsqueeze(0).expand(bs, 1, -1)
        targets = []
        with torch.no_grad():
            for t in range(num_bptt+1):
                next_inp = []
                next_inp.append(obs[:, t, :])
                next_inp = torch.cat([x.reshape(bs, -1) for x in next_inp], dim=1)
                target_outs, target_hidden = self.model(next_inp, target_hidden)
                targets.append(target_outs)
            targets = torch.stack(targets[1:], dim=1)
            assert targets.shape == (bs, num_bptt, self.args.n_actions)

            m, i = torch.max(targets, dim=2, keepdim=True)
            assert m.shape == (bs, num_bptt, 1)
            q_target = batch.r + self.gamma*(1-batch.d)*m
            assert q_target.shape == (bs, num_bptt, 1)

        Q_loss_elementwise = (q_values - q_target) ** 2
        Q_loss = torch.mean(Q_loss_elementwise * batch.m) / batch.m.sum() * np.prod(batch.m.shape)

        assert Q_loss.shape == ()

        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

    def store_episode(self, current_obs, action, reward, next_state, done, victim_action):
        self.buffer.push(current_obs, action, reward, next_state, done, False)

    def constraint_reward(self, z):
        return np.sum(np.array(z)*self.lambda_coef)

    def lambda_update(self, V, th):
        self.lambda_coef = self.lambda_coef - self.lambda_lr*(V-th/(1-self.gamma))

    def reset(self):
        self.buffer.reset()
        self.hidden = self.model.init_hidden()
        self.exploration_proba = 1.0
        self.constraint_value = []
