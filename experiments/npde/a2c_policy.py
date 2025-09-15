from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# This script defines the mode selector.
import modules as M


class RL_Trainer:
    def __init__(
        self,
        config,
        actors=None,
        critic=None,
        LR=None,
        other_params=[],  # noqa: B006
    ):
        self.critic = critic
        self.LR = LR
        self.steps = 0
        self.actors = actors

        self.lrate = config.lrate
        self.discount = config.discount
        self.beta_e = config.beta_e
        self.max_gradient = config.max_gradient
        self.MSE = torch.nn.MSELoss()

        # Collect parameters of all the modules
        params_dict = []
        if critic is not None:
            params_dict.append({"params": critic.parameters(), "lr": self.lrate})
        for actor in self.actors:
            actor.actor_log_probs = deque(maxlen=100000)
            actor.actor_entropies = deque(maxlen=100000)
            actor.actions = deque(maxlen=100000)
            params_dict.append({"params": actor.parameters(), "lr": self.lrate})
        for other_param in other_params:
            params_dict.append(other_param)

        self.params_dict = params_dict
        self.values = deque(maxlen=100000)
        self.rewards = deque(maxlen=100000)
        self.masks = deque(maxlen=100000)

        self.optimizer = optim.RMSprop(self.params_dict)
        self.train_log = {}

    def collect(
        self,
        actions,
        value,
        reward,
        done=False,
    ):
        if not hasattr(self.actors[0], "dist"):
            print("NO DIST: MODE SEL")
            return

        for actor, action in zip(self.actors, actions):
            actor.actor_log_probs.append(actor.dist.log_prob(action).sum().unsqueeze(0))
            actor.actor_entropies.append(actor.dist.entropy().mean().unsqueeze(0))
            actor.actions.append(action)

        self.values.append(value)
        self.rewards.append(torch.tensor([reward], dtype=torch.float))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float))

    def train(
        self,
    ):
        if not hasattr(self.actors[0], "dist"):
            return

        # Compute returns for value function with final value and all preceding rewards
        returns = self.compute_returns(
            list(self.values)[-1],
            list(self.rewards)[:-1],
            list(self.masks)[:-1],
            gamma=self.discount,
        )
        returns_vec = torch.cat(list(returns)).detach()
        values_vec = torch.cat(list(self.values))
        advantage = (returns_vec - values_vec)[:-1]

        critic_loss = advantage.pow(2).mean()
        loss = 0
        for j, actor in enumerate(self.actors):
            actor_log_probs_vec = torch.cat(list(actor.actor_log_probs))[:-1]
            actor_entropies_vec = torch.cat(list(actor.actor_entropies))[:-1]
            actor_loss = -(actor_log_probs_vec * advantage.detach()).mean() - self.beta_e * actor_entropies_vec.mean()
            loss = loss + actor_loss

        loss = loss + critic_loss

        # Step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_gradient / self.lrate)
        for actor in self.actors:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_gradient / self.lrate)
        self.optimizer.step()

        self.train_log["Critic loss"] = critic_loss.item()
        self.train_log["Actor entropy"] = actor_entropies_vec.mean().item()

    def compute_returns(self, next_value, rewards, masks, gamma=0.9):
        R = next_value
        returns = [R]
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def clear_history(self):
        for actor in self.actors:
            actor.actor_log_probs.clear()
            actor.actor_entropies.clear()
            actor.actions.clear()
        self.values.clear()
        self.rewards.clear()
        self.masks.clear()


# Models built from above components:
class A2C(nn.Module):
    def __init__(self, input_size, config, action_size=3):
        super().__init__()
        self.config = config
        self.layer_size = config.layer_size

        self.action_size = action_size
        self.data_size = input_size
        self.reward_size = 2
        self.feedback_size = self.action_size + self.reward_size
        self.input_size = self.feedback_size + self.data_size

        # Recurrence for main policy
        self.rnn = M.ESN(
            self.input_size,
            hidden_size=4096,
            recurrent_connection_probability=60 / 2048,
            input_connection_probability=0.2,
        )
        # self.rnn = M.Reservoir_Local(self.input_size, num_unique=40, num_shared=20)
        self.reservoir_size = self.rnn.hidden_size
        self.actor = M.Actor(self.reservoir_size, self.layer_size, self.action_size)
        self.critic = M.Critic(self.reservoir_size, self.layer_size)

        self.register_buffer("m_main", torch.ones(self.input_size))
        self.register_buffer("act_onehot", torch.eye(self.action_size))
        self.register_buffer("a_prev", self.act_onehot[0] * 0)
        self.register_buffer("rewards", torch.zeros(2))
        self.register_buffer("v", torch.zeros(1))
        self.a = 0
        self.m = 0

        # RL
        self.trainer = RL_Trainer(
            actors=[self.actor],
            critic=self.critic,
            config=self.config,
        )

    def forward(self, x):
        inputs = torch.cat([x, self.a_prev, self.rewards], -1)
        state = self.rnn(inputs).detach()
        self.a = self.actor(state)
        self.a_prev = self.act_onehot[self.a]
        self.v = self.critic(state)
        return self.a

    def init_hidden(self):
        self.rnn.h = self.rnn.h.detach() * 0
        self.a_prev = self.a_prev.detach() * 0
