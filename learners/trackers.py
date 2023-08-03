import copy

from modules.agents.rnn_agent import RNNAgent
from utils.buffer_recurrent import get_device, RecurrentReplayBuffer
import torch
import os
import numpy as np
import math


class TrackerAgent(RNNAgent):
    def __init__(self, input_shape, args):
        args_temp = copy.deepcopy(args)
        args_temp.use_rnn = True
        super(TrackerAgent, self).__init__(input_shape, args_temp)

    def forward(self, inputs, hidden_state):
        x = torch.nn.functional.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # q = torch.nn.functional.softmax(q, dim=-1)
        return q, h


class Tracker:
    def __init__(self, n_tracker_agents, input_shape, args):
        self.n_tracker_agents = n_tracker_agents
        self.input_shape = input_shape
        self.args = args
        self.agents = []
        self.agents_optimizers = []
        for agent in range(n_tracker_agents):
            self.agents.append(TrackerAgent(input_shape, args).to(get_device()))
            self.agents_optimizers.append(torch.optim.Adam(self.agents[agent].parameters(), lr=self.args.lr))

        self.buffer = RecurrentReplayBuffer((input_shape)*n_tracker_agents, 1, args.max_episode_length, #-1-args.n_actions
                                            args.buffer_size, args.batch_size)

        self.hidden = self.init_hidden()
        self.loss_function = torch.nn.NLLLoss(reduction='sum')

        self.last_reward = 0
        self.last_action = 0

        self.vic_act_prob_avg = []
        self.correct_predict_avg = []
        self.normality_score = np.array([])

        self.windows = self.args.tracker_window
        self.N_windows = len(self.windows)
        self.normality_metric = np.zeros((len(self.windows), self.n_tracker_agents))
        self.t = 0
        self.TH = args.detect_TH
        self.N_TH = len(self.TH)
        self.t_detect = 1000*np.ones((self.N_windows, self.N_TH, self.n_tracker_agents))

        self.out_dict = {"attacked":[], "t_start":[], "window_size": self.windows, "threshold": self.TH,
                         "t_detect": np.array([]), "battle_won":[], "ep_length":[]}

    def init_hidden(self):
        outs = []
        for agent in range(self.n_tracker_agents):
            outs.append(self.agents[agent].init_hidden().unsqueeze(0).expand(1, 1, -1))
        return outs

    def forward(self, inputs, rew, act, hidden_states):
        outs = []
        out_h = []
        obs_size = self.input_shape        #- 1 - self.args.n_actions
        for agent in range(self.n_tracker_agents):
            obs = inputs[agent*obs_size: (agent+1)*obs_size]
            inp = []
            inp.append(obs)
            #inp.append(torch.FloatTensor([rew]))
            #victim_last_action_vector = torch.zeros((1, self.args.n_actions))
            #victim_last_action_vector[:, act] = 1
            #inp.append(torch.FloatTensor(victim_last_action_vector))
            inp = torch.cat([x.reshape(1, -1) for x in inp], dim=1)
            q, h = self.agents[agent](inp, hidden_states[agent])
            q = torch.nn.functional.log_softmax(q, dim=-1)
            outs.append(q)
            out_h.append(h)
        return outs, out_h

    def train(self, agents_batch):
        bs, num_bptt = agents_batch.r.shape[0], agents_batch.r.shape[1]
        obs_size = self.input_shape        #- 1 - self.args.n_actions
        for agent in range(self.n_tracker_agents):
            obs = torch.FloatTensor(agents_batch.o[:, :, agent*obs_size: (agent+1)*obs_size])

            assert obs.shape == (bs, num_bptt+1, obs_size)

            predictions = []
            hidden = self.agents[agent].init_hidden().unsqueeze(0).expand(bs, 1, -1)
            loss = 0
            for t in range(num_bptt):
                inp = []
                inp.append(obs[:, t, :])
                inp = torch.cat([x.reshape(bs, -1) for x in inp], dim=1)
                agent_outs, hidden = self.agents[agent](inp, hidden)
                agent_outs = torch.nn.functional.log_softmax(agent_outs, dim=1)
                assert agent_outs.shape == (bs, self.args.n_actions)
                targets = torch.tensor(agents_batch.a[:, t]).long()
                loss += self.loss_function(agent_outs, torch.squeeze(targets))
                assert loss.shape == ()

            self.agents_optimizers[agent].zero_grad()
            loss.backward()
            self.agents_optimizers[agent].step()

    def save_trackers(self, save_dir):
        for agent in range(self.n_tracker_agents):
            save_name = "tracker_agent_{}.pth".format(agent)
            torch.save(self.agents[agent].state_dict(), os.path.join(save_dir, save_name))

    def load_trackers(self, save_dir):
        for agent in range(self.n_tracker_agents):
            save_name = "tracker_agent_{}.pth".format(agent)
            self.agents[agent].load_state_dict(
                torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device()))
            )

    def output_statistics(self, q, victim_action, reset=False):
        if reset:
            self.out_dict["t_detect"] = np.append(self.out_dict["t_detect"], self.t_detect)
            self.out_dict["t_detect"] = np.reshape(self.out_dict["t_detect"], (-1, self.N_windows, self.N_TH,
                                                                               self.n_tracker_agents))
            #self.out_dict["final_normality"].append((self.normality_metric[0][:]/self.t).tolist())
            self.normality_metric = np.zeros((len(self.windows), self.n_tracker_agents))
            self.normality_score = np.array([])
            self.t = 0
            self.t_detect = 1000*np.ones((self.N_windows, self.N_TH, self.n_tracker_agents))

            return
        else:

            self.t += 1
            self.normality_score = np.append(self.normality_score, np.zeros(self.n_tracker_agents))
            self.normality_score = np.reshape(self.normality_score, (-1, self.n_tracker_agents))
            for agent in range(self.n_tracker_agents):
                q_a = q[agent]
                m, i = torch.max(q[agent], dim=1)
                self.normality_score[-1][agent] = (q_a[0][victim_action].tolist() - torch.squeeze(m).tolist())
                for w in range(self.N_windows):
                    ws = self.windows[w]
                    if ws == -1:
                        self.normality_metric[w][agent] += self.normality_score[-1][agent]
                        metric = self.normality_metric[w][agent] / self.t
                        for th in range(self.N_TH):
                            if metric < self.TH[th] and self.t_detect[w][th][agent] > self.args.max_episode_length:
                                self.t_detect[w][th][agent] = self.t
                    else:
                        if self.t <= ws:
                            self.normality_metric[w][agent] = self.normality_metric[0][agent]
                            for th in range(self.N_TH):
                                self.t_detect[w][th][agent] = self.t_detect[0][th][agent]
                        else:
                            self.normality_metric[w][agent] += (self.normality_score[-1][agent]
                                                                - self.normality_score[-1-ws][agent])
                            metric = self.normality_metric[w][agent] / ws
                            for th in range(self.N_TH):
                                if metric < self.TH[th] and self.t_detect[w][th][agent] > self.args.max_episode_length:
                                    self.t_detect[w][th][agent] = self.t


    def output_results(self):
        battle_win = self.out_dict["battle_won"]
        episode_len = self.out_dict["ep_length"]
        #w = 0 if self.args.tracker_window == -1 else 1
        for w in range(self.N_windows):
            ttd = []
            pr = []
            wr = []
            window = "no window" if w == 0 else "{}".format(self.windows[w])
            for th_ind in range(len(self.TH)):
                n_positive = 0
                n_eps = 0
                t_sum = 0
                won_matches = 0
                for eps in range(len(battle_win)):
                    if np.min(self.out_dict["t_detect"][eps][w][th_ind][1:]) >= self.out_dict["t_start"][eps]:
                        n_eps += 1
                        if np.min(self.out_dict["t_detect"][eps][w][th_ind][1:]) < 1000:
                            t_sum += np.min(self.out_dict["t_detect"][eps][w][th_ind][1:]) - self.out_dict["t_start"][eps]
                            n_positive += 1
                        else:
                            t_sum += episode_len[eps] - self.out_dict["t_start"][eps]
                        if battle_win[eps] == 1:
                            won_matches += 1

                ttd.append((t_sum / n_eps) if n_eps != 0 else -1)
                pr.append((n_positive / n_eps) if n_eps != 0 else -1)
                wr.append((won_matches/n_eps) if n_eps != 0 else -1)

            print("##### Results Summary ##########################################")
            print("Complete Episodes:{}".format(n_eps))
            print("Window Size:" + window)
            print("Thresholds:{}".format(self.TH))
            print("Positive Rate:{}".format(pr))
            print("Time to Detect:{}".format(ttd))
            print("#################################################################")

