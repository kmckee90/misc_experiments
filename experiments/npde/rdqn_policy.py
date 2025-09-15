from copy import deepcopy

import torch
import torch.nn as nn

import modules as M


class CircularBuffer:
    def __init__(self, size, capacity, dtype=torch.float):
        self.capacity = capacity
        self.buffer = torch.zeros(self.capacity, size, dtype=dtype)
        self.idx = 0
        self.wrapped = False

    def append(self, x):
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.wrapped = True

    def last(self):
        return self.buffer[(self.idx - 1) % self.capacity]

    def __call__(self, *args, **kwds):
        if not self.wrapped:
            return self.buffer[: self.idx]
        else:
            return self.buffer


class Agent(nn.Module):
    def __init__(self, obs_size, action_size, **kwargs):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = kwargs["gamma"]
        self.n_optimizer_steps = kwargs["n_optimizer_steps"]
        self.lr = kwargs["learning_rate"]
        self.target_update_interval = kwargs["target_update_interval"]

        # self.rnn = M.Reservoir_Local(obs_size + self.action_size + 2, 40, 20)

        self.rnn = M.ESN(
            obs_size + self.action_size + 2,
            hidden_size=4096,
            recurrent_connection_probability=60 / 4096,
            input_connection_probability=0.1,
        )

        self.reservoir_size = self.rnn.hidden_size

        self.layer_size = kwargs["layer_size"]
        self.model = nn.Sequential(
            nn.Linear(self.reservoir_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.action_size),
        )
        self.target_model = deepcopy(self.model)
        with torch.no_grad():
            for param in self.target_model.parameters():
                param.data.zero_()

        self.register_buffer("actions", torch.eye(self.action_size))
        self.memory_length = kwargs["memory_length"]
        self.minibatch_size = kwargs["minibatch_size"]

        self.mem_reward = CircularBuffer(1, self.memory_length, dtype=torch.float32)
        self.mem_action = CircularBuffer(1, self.memory_length, dtype=torch.int64)
        self.mem_state = CircularBuffer(self.rnn.h.shape[-1], self.memory_length, dtype=torch.float32)
        self.mem_done = CircularBuffer(1, self.memory_length, dtype=torch.bool)

        self.train_log = {}

    def act_epsilon(self, x):
        self.forward(x)
        return torch.randint(0, self.action_size, (1,)).item()

    def act_policy(self, x):
        return self.forward(x).argmax().item()

    def remember(self, action, reward, done):
        self.mem_action.append(action)
        self.mem_reward.append(reward)
        self.mem_state.append(self.rnn.h.clone().detach())
        self.mem_done.append(done)

    def clear_memory(self):
        self.mem_reward = CircularBuffer(1, self.memory_length, dtype=torch.float32)
        self.mem_action = CircularBuffer(1, self.memory_length, dtype=torch.int64)
        self.mem_state = CircularBuffer(self.rnn.h.shape[-1], self.memory_length, dtype=torch.float32)
        self.mem_done = CircularBuffer(1, self.memory_length, dtype=torch.bool)

    # Hard update method:
    def update(self, utilities=None):
        if self.mem_state.idx < 10 and not self.mem_state.wrapped:
            return
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        valid_idx = torch.arange(len(self.mem_done()))  #####
        valid_idx = valid_idx[valid_idx != (self.mem_state.idx - 1) % self.mem_state.capacity]

        if utilities is not None:
            valid_idx = valid_idx[utilities[valid_idx] == 1]

        for step in range(self.n_optimizer_steps):
            minibatch_size = min(self.minibatch_size, len(self.mem_reward()))
            minibatch_idx = valid_idx[torch.randperm(len(valid_idx))[:minibatch_size]]

            with torch.no_grad():
                states = self.mem_state()[minibatch_idx]
                actions = self.mem_action()[minibatch_idx]
                rewards = self.mem_reward()[minibatch_idx]
                next_states = self.mem_state()[(minibatch_idx + 1) % self.mem_state.capacity]
                dones = self.mem_done()[minibatch_idx]

            q_values_next = self.target_model(next_states).detach()
            max_q_values_next = torch.max(q_values_next, dim=1).values.unsqueeze(1)
            targets = rewards + self.gamma * max_q_values_next * ~dones
            q_values = self.model(states.detach())
            q_values = q_values.gather(1, actions)

            mse = torch.nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()

            if step % self.target_update_interval == 0 and step > 0:
                pars = self.model.state_dict()
                self.target_model.load_state_dict(pars)

        self.train_log = {"MSE": mse.item()}

    def forward(self, x):
        rew = self.mem_reward.last()
        x = torch.cat(
            [
                x,
                self.actions[self.mem_action.last()].squeeze(),
                torch.sign(rew),
                rew,
            ]
        )
        z = self.rnn(x)
        a = self.model(z)
        return a

    def reset_models(self):
        self.model = nn.Sequential(
            nn.Linear(self.reservoir_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.action_size),
        )
        self.target_model = deepcopy(self.model)
        with torch.no_grad():
            for param in self.target_model.parameters():
                param.data.zero_()
