# Cartpole-v1
It is just a python script to perform the standard cartpole problem(open ai gym) using a Deep Q Network.

NEURAL NETWORK ARCHITECTURE

I have used a Q network with three layers.
The first layer contains 256nodes and the second layer contains 256 nodes and the third layer contains 2 nodes(the third layer is output layer and the action space of cartpole is 2 either left or right).
The first layer takes the state space and maps into the 256 nodes which is the input layer and the second layer is the hidden layer.

REPLAY BUFFER

I made the code to save upto 10,000 transition tuples to train my neural network, since without buffer myv neural network would start overfiiting to current set of actions which would drastically change the weights and would fail for any different state. I create a deque which could hold upto 10,000 transition tuple out of which using random i pick a small batch of 64 transition tuple and train my q network accordingly.And also since my transition tuple are numpy arrays i need to convert them into pytorch tensors to feed into pytorch q network.

I have used two Q network where one network is active and the other is a stable network which is built by this buffer and ti update this stable or target network i used a soft update technique.

To compromise the exploration exploitation trade off i have used epsilon greedy strategy.

