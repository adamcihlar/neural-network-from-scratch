### About
Neural network implemented from scratch in C++. \
The task was to load data, train network, evaluate and export results within 30 minutes and reach 88 % accuracy on the test set.
A standard feedforward neural network is implemented with various generalization and optimization improvements - Nesterov momentum, dropout, weight decay, learning rate decay, etc.

#### Running the code
* By default the code runs on 16 cores - set in the beginning of the source code
* ./RUN compiles and executes the solution
* The solution:
	1. loads data
	2. trains neural network
	3. performs inference on test set
	4. saves predicted labels from training and test set
* The results (accuracy on train and test) are displayed in the terminal
