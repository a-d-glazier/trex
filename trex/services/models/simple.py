import torch

class GailNet(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = False
        shape = 22 # just the size of the next state
        hidden_dim = 64
        self.fc1 = torch.nn.Linear(shape, hidden_dim) # use the shape? 
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, state, action, next_state, done=None):
        # argmax_index = torch.argmax(action, dim=1)

        # # Create a one-hot vector based on the argmax index
        # action = torch.zeros_like(action, dtype=torch.float32)
        # action.scatter_(1, argmax_index.unsqueeze(1), 1.0)

        x = torch.cat((state, action, next_state), dim=1)
        # x = next_state
        
        x = torch.nn.functional.relu(self.norm1(self.fc1(x)))
        x = torch.nn.functional.relu(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        
        return x.squeeze(-1) #if state.shape[0] == 1 else x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16) # use the shape? right now 3 is (s,a,s')
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            # x = traj.view(-1, 3) # use the shape?
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            r = x
            # r = self.softmax(x)
            sum_rewards += torch.sum(r)
            sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j