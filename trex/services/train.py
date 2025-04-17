import random
import torch
import numpy as np

def create_training_data(ranked_trajectories):
    """
    Create training data from the ranked trajectories.

    Args:
        ranked_trajectories: A list of tuples `(trajectory, score)` where
            `score` is the result of the ranking function applied to the
            trajectory.

    Returns:
        A tuple `(X, y)` where `X` is a list of feature vectors and `y` is
        a list of target values.
    """
    X = []
    y = []

    for n in range(len(ranked_trajectories)):
        # pick two random trajectories
        traj_i = random.choice(ranked_trajectories)
        rank_i = ranked_trajectories.index(traj_i)
        actual_return_i = ranked_trajectories[rank_i][1]
        transitions_i = traj_i[0].transitions()

        traj_j = random.choice(ranked_trajectories)
        rank_j = ranked_trajectories.index(traj_j)
        actual_return_j = ranked_trajectories[rank_j][1]
        transitions_j = traj_j[0].transitions()

        # determine the label
        # label = 1 if rank_i < rank_j else 0 # works because we're assuming ranked is sorted best to worst
        if actual_return_i > actual_return_j:
            label = 1
        elif actual_return_i < actual_return_j:
            label = 0
        else:
            continue # skip if they have the same return

        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        # min_length = min(len(transitions_i), len(transitions_j))
        # rand_length = random.randint(1, min_length)
        # if actual_return_i < actual_return_j: #pick tj snippet to be later than i
        #     ti_start = np.random.randint(min_length - rand_length + 1) 
        #     tj_start = np.random.randint(ti_start, len(transitions_j) - rand_length + 1)
        # else: #ti is better so pick later snippet in ti
        #     tj_start = np.random.randint(min_length - rand_length + 1)
        #     ti_start = np.random.randint(tj_start, len(transitions_i) - rand_length + 1)

        clipped_i = transitions_i#[ti_start:ti_start+rand_length]
        clipped_j = transitions_j#[tj_start:tj_start+rand_length]

        # TODO: implement partial trajectory slicing

        # Add to dataset
        X.append((clipped_i, clipped_j))
        y.append(label)

    return X, y

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    loss_criterion = torch.nn.CrossEntropyLoss()
    
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")