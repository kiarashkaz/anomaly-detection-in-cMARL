from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import copy
import learners.single_agent_dqn
import time


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.adv_active = self.args.attack_active
        self.attack_start_t = np.random.randint(0,
                                                high=self.args.attack_max_start)\
            if self.args.attack_start_t == -1 else self.args.attack_start_t

        self.episode_result = 0
        self.episode_len = 0


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.attack_start_t = np.random.randint(0,
                                                high=self.args.attack_max_start) if self.args.attack_start_t == -1 else self.args.attack_start_t

    def run(self, advagent=None, tracker=None,test_mode=False):
        self.reset()
        if self.args.save_replay:
            self.env.render()
        adv_test_mode = advagent.test_mode if self.adv_active else True
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        tracker.hidden = tracker.init_hidden()
        if self.adv_active:
            advagent.hidden = advagent.model.init_hidden()
        tracker.last_reward = 0
        tracker.last_action = 0
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            trackers_obs = []
            for agent in range(tracker.n_tracker_agents):
                trackers_obs = [*trackers_obs, *pre_transition_data["obs"][0][agent]]
            trackers_obs = th.tensor(trackers_obs)
            ###################################################################################
            if self.args.attack_active:
                if self.args.attack_type == "OA":
                    #adv_state = pre_transition_data["obs"][0]
                    adv_state = pre_transition_data["obs"][0][0]
                    adv_state = np.array(adv_state)
                    adv_avail_actions = self.env.get_avail_agent_actions(0)
                    if self.adv_active and self.t >= self.attack_start_t:
                        adv_action = advagent.compute_action(adv_state, adv_avail_actions)
                        input_obs = copy.deepcopy(pre_transition_data["obs"])
                        batch_temp = copy.deepcopy(self.batch)
                        victim_perturbed_obs = self.jsma_perturb(batch_temp, advagent.input_shape, adv_action,
                                                                 input_obs,
                                                                 theta=0.5, max_iter=10)
                        pre_transition_data["obs"][0][0] = victim_perturbed_obs
                        self.batch.update(pre_transition_data, ts=self.t)
                else:
                    adv_state = pre_transition_data["obs"][0][0]
                    adv_state = np.array(adv_state)
                    adv_avail_actions = self.env.get_avail_agent_actions(0)
                    if self.adv_active and self.t >= self.attack_start_t:
                        adv_action = advagent.compute_action(adv_state, adv_avail_actions)

            ####################################################################################
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            victim_action = actions[0][0]
            #####################################
            if self.adv_active and self.t >= self.attack_start_t and self.args.attack_type != "OA":
                actions[0][0] = adv_action
                victim_action = actions[0][0]
            # if not advagent.test_mode:
            #   actions[0][0] = adv_action
            #  victim_action = actions[0][0]
            ######################################
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            if self.args.save_replay:
                self.env.render()
                time.sleep(0.5)

            if tracker.args.train:
                tracker.buffer.push(trackers_obs, victim_action, reward, 0, terminated, False)
            else:
                q, tracker.hidden = tracker.forward(trackers_obs, tracker.last_reward, tracker.last_action, tracker.hidden)
                tracker.output_statistics(q, victim_action)
                tracker.last_reward = reward
                tracker.last_action = victim_action
                if not adv_test_mode:
                    adv_reward = - reward
                    #nscore = np.average(tracker.normality_score[-1, 1:])
                    z = tracker.normality_score[-1, 1:]
                    adv_reward += advagent.constraint_reward(z)
                    next_obs = [self.env.get_obs()]
                    adv_next_state = next_obs[0][0]
                    adv_next_state = np.array(adv_next_state)
                    advagent.store_episode(adv_state, adv_action, adv_reward, adv_next_state, terminated, victim_action)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        #print("z is: {}".format(tracker.normality_score))
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        self.episode_result = cur_stats['battle_won'] if 'battle_won' in cur_stats else -1
        self.episode_len = self.t
        # if not test_mode:
        #    self.t_env += self.t
        self.t_env += self.t
        cur_returns.append(episode_return)

        if test_mode and not tracker.args.train and adv_test_mode: #and (len(self.test_returns) == self.args.test_nepisode-1):
            self._log(cur_returns, cur_stats, log_prefix)
        elif test_mode and (len(self.test_returns) == self.args.test_nepisode-1):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def jsma_perturb(self, batch, adv_n_features, target_action, input_obs, theta=0.5, max_iter=5, norm_limit=10):

        if theta == -1:    # theta = -1 ==> dynamic
            theta_min = 0.1
            theta_max = 0.9
            theta_step = (theta_max - theta_min)/(max_iter-1)
            theta_vector = np.arange(theta_min, theta_max+theta_step, theta_step)
        else:
            theta_vector = np.ones(max_iter)*theta

        adv_obs = copy.deepcopy(input_obs[0][0])
        agents_obs = input_obs
        perturbed_obs = input_obs[0][0]   # input_obs => all agents
        it = 0
        #batch = copy.deepcopy(self.batch)
        t_ep = copy.deepcopy(self.t)
        t_env = copy.deepcopy(self.t_env)
        actions = self.mac.select_actions(batch, t_ep, t_env, test_mode=True, temp_mode=True)
        victim_action = actions[0][0]
        target_achieved = (victim_action == target_action)
        limit_reached = False
        while it < max_iter and not target_achieved and not limit_reached:
            #batch = copy.deepcopy(self.batch)
            agents_q_values = self.mac.q_values_calc(batch, t_ep, t_env, test_mode=True, temp_mode=True)
            victim_q_values = agents_q_values[0][0]
            target_delta_q = []        # delta_q => [feature]
            non_target_delta_q = []

            for k in range(adv_n_features) :
                temp_obs = copy.deepcopy(perturbed_obs)
                temp_obs[k] += 0.01 #theta_vector[it]
                agents_obs[0][0] = temp_obs
                #batch = copy.deepcopy(self.batch)
                batch.update({"obs": agents_obs}, ts=t_ep)
                perturbed_agents_q = self.mac.q_values_calc(batch, t_ep, t_env, test_mode=True, temp_mode=True)
                perturbed_q = perturbed_agents_q[0][0]
                feature_delta_q = perturbed_q[target_action] - victim_q_values[target_action]
                target_delta_q.append(feature_delta_q)
                feature_delta_q_sum = (th.sum(perturbed_q[perturbed_q != -float("inf")]) -
                                         th.sum(victim_q_values[victim_q_values != -float("inf")]))
                non_target_delta_q.append(feature_delta_q_sum-feature_delta_q)

            best_criterion = 0
            target_i = 0
            target_j = 0
            for i in range(adv_n_features-1):
                target_i_term = target_delta_q[i]
                non_target_i_term = non_target_delta_q[i]
                for j in range(i+1,adv_n_features):
                    target_j_term = target_delta_q[j]
                    non_target_j_term = non_target_delta_q[j]
                    criterion = (target_i_term + target_j_term)*(non_target_i_term + non_target_j_term)
                    if criterion < 0 :
                        if -criterion > best_criterion :
                            best_criterion = -criterion
                            target_i = i
                            target_j = j
            if best_criterion > 0:
                perturbed_obs[target_i] += theta_vector[it] * th.sign(target_delta_q[target_i] + target_delta_q[target_j])
                perturbed_obs[target_j] += theta_vector[it] * th.sign(target_delta_q[target_i] + target_delta_q[target_j])
            agents_obs[0][0] = perturbed_obs
            #batch = copy.deepcopy(self.batch)
            batch.update({"obs": agents_obs}, ts=t_ep)
            actions = self.mac.select_actions(batch, t_ep, t_env, test_mode=True, temp_mode=True)
            victim_action = actions[0][0]
            target_achieved = (victim_action == target_action)
            if np.linalg.norm(perturbed_obs-adv_obs, 1) > norm_limit:
                limit_reached = True
            it += 1

        return perturbed_obs
