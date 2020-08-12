import torch
import torch.nn as nn
import numpy as np

'''
6 X 6 (= 36) world where there are 36 * 4 (= 144) possible colour predictions for 4 directions

Out of these, 6 X 4 (= 24) directions are non-white and 144 - 24 (= 120) directions are white

Two baselines:
A) All predictions are white					   
B) All predictions have the same probability	   
C) All predictions are always wrong				
D) All predictions are always correct			  
E) All predictions are zero (initial weights are zeros, so all predictions before learning happens are zeros)
---------------------------------------------------------------------------------------

For case A)
Prediction			 [1, 0, 0, 0, 0, 0]

White observation	  [1, 0, 0, 0, 0, 0]	 MSE = 0
Non White observation  [0, 0, 1, 0, 0, 0]	 MSE = 1/3

Total loss based on state visitation = ( 0 * 120 + 1/3 * 24 ) / 144 = 8 / 144 = 0.0555
Total loss based on colour = (0 * 1/6 + 1/3 * 5/6) = 0.3999
---------------------------------------------------------------------------------------

For case B)
Prediction			 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

White observation	  [1, 0, 0, 0, 0, 0]	 MSE = 0.1389
Non White observation  [0, 0, 1, 0, 0, 0]	 MSE = 0.1389

Total loss based on state visitation = ( 0.1389 * 120 + 0.1389 * 24 ) / 144 = 40 / 144 = 0.1389
Total loss based on colour = (0.1389 * 1/6 + 0.1389 * 5/6) = 0.1389

---------------------------------------------------------------------------------------

For case C)
Prediction			 [0, 0, 0, 0, 0, 1]

White observation	  [1, 0, 0, 0, 0, 0]	 MSE = 1/3
Non White observation  [0, 0, 1, 0, 0, 0]	 MSE = 1/3

Total loss based on state visitation = ( 1/3 * 120 + 1/3 * 24 ) / 144 = 48 / 144 = 0.3333
Total loss based on colour = (1/3 * 1/6 + 1/3 * 5/6) = 0.3333

---------------------------------------------------------------------------------------

For case D)
Prediction			 [0, 1, 0, 0, 0, 0]

Observation			[0, 1, 0, 0, 0, 0]	 MSE = 0

Total loss based on state visitation = ( 0 * 120 + 0 * 24 ) / 144 = 0 / 144 = 0.0
Total loss based on colour = (0 * 1/6 + 0 * 5/6) = 0.0

---------------------------------------------------------------------------------------

For case E)
Prediction			 [0, 0, 0, 0, 0, 0]

Observation			[0, 1, 0, 0, 0, 0]	 MSE = 1/6

Total loss based on state visitation = ( 1/6 * 120 + 1/6 * 24 ) / 144 = 0.1667
Total loss based on colour = (1/6 * 1/6 + 1/6 * 5/6) = 0.1667



'''

class RNNState(nn.Module):
	def __init__(self):
		super(RNNState, self).__init__()

		'''
		Initialize the parameters for the RNN state representation
		'''

		self.input_size = 6+6+6
		self.hiddenstate_size = 256
		self.number_recurrent_layers = 2
		self.dropout = 0.0
		self.nonlinearity = 'tanh'
		self.batch_first = False
		self.bias = True
		self.bidirectional = False
		self.num_directions = 1
		self.seq_len = 1
		self.batch = 1

		self.output_size = 6
		self.hiddenstate = torch.zeros(self.number_recurrent_layers * self.num_directions,
									   self.batch, self.hiddenstate_size)
		
		self.rnn = nn.RNN(input_size = self.input_size , hidden_size = self.hiddenstate_size, 
						  num_layers = self.number_recurrent_layers, bias = self.bias,
						  batch_first = self.batch_first, dropout = self.dropout,
						  nonlinearity = self.nonlinearity, bidirectional = self.bidirectional)
		
		self.fc = nn.Linear(self.hiddenstate_size, self.output_size)

		'''
		Predictions are positive, but using relu does not allow for
		negative TDE errors and leads to no learning
		#self.relu = nn.ReLU()
		'''


	def forward(self, old_observation, action):
		'''
		Transform the input to the state representation using observation and action
		'''

		if torch.all( torch.eq( action, torch.tensor([1, 0, 0], dtype=torch.float) ) ):
			input = torch.cat((old_observation, torch.zeros(6, dtype=torch.float)))
			input = torch.cat((input, torch.zeros(6, dtype=torch.float)))
		elif torch.all( torch.eq( action, torch.tensor([0, 1, 0], dtype=torch.float) ) ):
			input = torch.cat((torch.zeros(6, dtype=torch.float), old_observation))
			input = torch.cat((input, torch.zeros(6, dtype=torch.float)))
		elif torch.all( torch.eq( action, torch.tensor([0, 0, 1], dtype=torch.float) ) ):
			input = torch.cat((torch.zeros(6, dtype=torch.float), torch.zeros(6, dtype=torch.float)))
			input = torch.cat((input, old_observation))
			
		
		'''
		Forward pass through the state representation
		'''

		input = input.view(self.seq_len, self.batch, self.input_size)
		self.output, self.hiddenstate = self.rnn(input, self.hiddenstate)
		self.hiddenstate = self.hiddenstate.detach()
		self.output = self.output.view(self.hiddenstate_size)
		self.output = self.fc(self.output)
		self.prediction = self.output.view(1,6)
		return self.prediction


class RNNAgent():
	def __init__(self, gamma):
		self.gamma = gamma

	def start(self, observation):
		'''
		Agent starts with some observation and returns an action

		Initializes parameters for the agent
		'''

		self.timesteps = 0
		self.flag = False
		self.total_loss_list = np.array([])
		self.total_loss_list_white = np.array([])
		self.total_loss_list_nonwhite = np.array([])
		self.display_length = 1000

		'''
		Initializes the state representation for the agent
		
		Initializes the loss function for the agent
		
		Initializes the optimizer for the agent
		'''

		self.model = RNNState()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

		self.criterion = nn.CrossEntropyLoss()

		'''
		Agent returns an action
		
		New observation becomes old observation as agent steps into time
		'''		

		self.new_observation = torch.tensor(observation, dtype=torch.float).view(6)
		self.action = torch.tensor([0, 0, 0], dtype=torch.float)
		index = torch.randint(0, 3, (1,))[0]
		self.action[index] = 1
		self.old_observation = self.new_observation

		return self.action.tolist()


	def step(self, observation, term):
		'''
		Agent receives an observation and a reward for the action it took earlier
		
		It returns another action based on this observation and reward
		'''

		self.timesteps += 1
		if self.timesteps % 1000 == 0:
			self.flag = True

		self.new_observation = torch.tensor(observation, dtype=torch.float).view(6)

		'''
		Agent learns by forward pass and backward pass through backpropagation
		and returns the loss
		'''

		loss = self.train(self.old_observation, self.new_observation, self.action, term)

		'''
		Printing logs
		'''

		if self.flag == True:
			totalloss_mean = np.mean(self.total_loss_list)
			totalloss_white_mean = np.mean(self.total_loss_list_white)
			totalloss_nonwhite_mean = np.mean(self.total_loss_list_nonwhite)

			# weighted mean based based on probability of state distribution of policy
			#totalloss = np.sum((120.0 / 144)  * self.total_loss_list_white / len(self.total_loss_list)) + np.sum((24.0 / 144) * self.total_loss_list_nonwhite / len(self.total_loss_list))
			
			# weighted mean based on colours
			totalloss_mean_colour = 1.0 / 6 * totalloss_white_mean + 5.0 / 6 * totalloss_nonwhite_mean
			print("Timestep: ", self.timesteps)
			print("Total Loss based on state visitation: ", totalloss_mean)
			print("Total Loss based on colour: ", totalloss_mean_colour)
			print("Total Loss on White: ", totalloss_white_mean)
			print("Total Loss on Non White: ", totalloss_nonwhite_mean)
			self.total_loss_list = np.array([])
			self.total_loss_list_white = np.array([])
			self.total_loss_list_nonwhite = np.array([])
			print('-------------------------------')
			self.flag = False
		
		'''
		Agent returns an action
		
		New observation becomes old observation as agent steps into time
		'''

		self.action = torch.tensor([0, 0, 0])
		index = torch.randint(0, 3, (1,))[0]
		self.action[index] = 1
		self.old_observation = self.new_observation
		
		return self.action.tolist()
	
	
	def train(self, old_observation, new_observation, action, term):
		'''
		Forward and backward pass
		
		Forward pass to calculate prediction of value of the old state-action
		
		Calculates loss as TD error with target as sum of reward
		and discounted expected value of next state-actions 
		'''
		
		prediction = self.model.forward(old_observation, action)
		'''
		Expected update over policy = 0.5 probability
		
		Can also do it using Q-learning updates by taking max over the two values and
		using that as the target
		'''

		'''
		Can set gamma = 0 on encountering state with observation and reward = 1
		to indicate end of prediction

		Not setting gamma = 0 does not lead to convergence in predictions
		'''
	
		

		'''
		Calculates TDE loss
		'''
		
		loss = self.backward(new_observation, prediction)
		
		'''
		Printing logs
		'''
		if new_observation[0] == 1:	
			self.total_loss_list_white = np.append(self.total_loss_list_white, loss)
			#print('White Loss: ', loss)
			#print('Observation: ', new_observation)
			#print('Prediction: ', prediction)

		else:
			self.total_loss_list_nonwhite = np.append(self.total_loss_list_nonwhite, loss)
			#print('Non White Loss: ', loss)
			#print('Observation: ', new_observation)
			#print('Prediction: ', prediction)
		self.total_loss_list = np.append(self.total_loss_list, loss)

		return loss
	
	
	def backward(self, new_observation, prediction):
		'''
		Backward pass through backpropagation using SGD
		
		Calculates TD error
		
		Loss function is Squared TD error
		'''
		new_observation = new_observation.long()
		self.loss = self.criterion(prediction, (new_observation == 1).nonzero()[0] )		

		loss = self.loss.item()
		
		if self.timesteps > 1000:
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()

		return loss