# Continuous Control Project Report

## Learning Algorithm

The Learning Algorith is based on the the [Deep Determanistic Policy Gradient (DDPG) implementation](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py). In this algorithm, it will be used a **Replay Buffer**, a memory of BUFFER_SIZE = 1e5 cells, where it will be stored tuple made of:
```
(state,action,reward,next_state, episode_end)
```
If are available enough examples into Replay Buffer (more than batch size), every **n_step** (set to 1) and for **n_times** (set to 2) it will be extracted (dequed) a random batch of examples to be used for network weights updates, the so called learning phase.

In this algorithm we merge the policy and values-based approach, usually named **Critic/Actor**:
- a Critic measures how good the action taken is (value-based)
- an Actor controls how our agent behaves (policy-based)

2 Neural Networks, that is a set of θ (weights), for each Actor/Critic network will be used, defined as follow:

**Actor**:

- Input layer: 33
- fully connected layer: units=128
- activation: ReLU
- batch normalization layer
- fully connected layer: units=128
- activation: ReLU
- fully connected: 4
- Output activation: tanh()

**Critic**:

- Input layer: 33
- fully connected layer: units=128
- activation: ReLU
- batch normalization layer
- fully connected layer: units=128 + 4 (actions)
- activation: ReLU
- Output fully connected: 1


To update weights, it will be used this formula:
```
Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
```
where:
- actor_target(state) -> action
- critic_target(state, action) -> Q-value

Params:
- experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
- gamma (float): discount factor

Note: this is the reason why the second fully connected layer into Critic Network takes in input a tensor with action, the actor_target() output. 

The final update is done using the "local" copy of actor/critic network weights, with a so called "soft update" to the "target" copy of actor/critic network weights, following this formula:
```
θ_target = τ*θ_local + (1 - τ)*θ_target
```

Training optimizer used it's an Adam with a learning rate of 2e-4 for both networks, with a weight decay of 0 and a batch size of 128.

For the environment has been used a single Agent, that reached the target average score of 30 over 100 consecutive episodes after 347 episodes!

The progress chart has been:
<p align="left">
  <img src="running.png" width="350" title="hover text">
</p>

The Agent, before training, runs as in this video:

<p align="left">
  <img src="arm-not-training.gif" width="400" title="hover text">
</p>

After Training, the agent runs as follows:

<p align="left">
  <img src="arm-trained.gif" width="400" title="hover text">
</p>


## Further Improvement

Several improvements might be done to increase the network perfomances, as:
* reduce variance
* use 20 agents in parallel to exploit better DDPG algorithm
* try batch normalization on each hidden layer
* larger hidden layers
