import torch
import torch.nn as nn
import numpy as np
import pickle
import os

class RNNAgent():
	def __init__(self, gamma, truncation_length, learning_rate):
		self.gamma = gamma
		self.truncation_length = truncation_length
		self.learning_rate = learning_rate

	def start(self, observation, run):
		'''
		Agent starts with some observation and returns an action. Initializes parameters for the agent.
		Sets random seed
		'''

		np.random.seed(run)
		torch.manual_seed(run)

		self.finallosslist = np.array([])
		self.finalobservationslist = np.array([])

		self.display_length = 1000

		self.observation_buffer = torch.tensor([], dtype=torch.float)
		self.seq_len = self.truncation_length


		'''
		Initializes the state representation, loss function for the agent, and the optimizer for the agent.
		'''

		self.model = RNNState(input_size=1, hiddenstate_size=14, number_recurrent_layers=1,
		                    dropout=0.0, nonlinearity='tanh', batch_first=False, bias=True,
							bidirectional=False, num_directions=1, seq_len=self.seq_len, batch=1, output_size=1)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

		'''
		Agent returns an action. New observation becomes old observation as agent steps into time
		'''		

		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation

		self.observation_buffer = torch.cat([self.observation_buffer, self.old_observation])

		return self.action


	def step(self, observation, reward, end, run):
		'''
		Agent receives an observation and a reward for the action it took earlier
		It returns another action based on this observation and reward
		'''
		
		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		
		
		if len(self.observation_buffer) == self.seq_len:

			self.new_observation_buffer = torch.cat([self.observation_buffer, self.new_observation])[-self.seq_len:]
			
			'''
			Agent learns by forward pass and backward pass through backpropagation
			and returns the loss
			'''

			loss = self.train(self.observation_buffer, self.new_observation_buffer, self.action)

			'''
			Agent returns an action		
			New observation becomes old observation as agent steps into time
			'''


		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation
		self.observation_buffer = torch.cat([self.observation_buffer, self.old_observation])[-self.seq_len:]
		
		if end == True:
			dirpath = 'Data/RingWorldRNNData/'
			if not os.path.exists(dirpath):
				os.makedirs(dirpath)

			pickle.dump(self.finallosslist, open(dirpath + 'run'+str(run)+'_lossSimple','wb'))
			pickle.dump(self.finalobservationslist, open(dirpath+'run'+str(run)+'_observationsSimple','wb'))

		return self.action
	

	def getInput(self, observation_buffer, action):
		'''
		Transform the input to the state representation using observation and action
		'''

		input = torch.tensor([])
		for i in range(len(observation_buffer)):
			temp = torch.tensor([])
			
			if observation_buffer[i] == torch.tensor(0):
				temp = torch.tensor([1,0], dtype=torch.float).view(2)
			elif observation_buffer[i] == torch.tensor(1):
				temp = torch.tensor([0,1], dtype=torch.float).view(2)
			
			if action == 0:
				temp = torch.cat([temp, torch.tensor([0, 0], dtype=torch.float)])
			elif action == 1:
				temp = torch.cat([torch.tensor([0, 0], dtype=torch.float), temp])

			input = torch.cat([input, temp])

		return input


	def train(self, observation_buffer, new_observation_buffer, action):
		'''
		Forward pass to calculate prediction of value of the old state-action
		Backward pass for n-step BPTT
		Calculates loss as TD error with target as sum of cumulant
		and discounted expected value of next state-actions 
		'''
		
		self.value_old_state_action = self.model.forward(self.getInput(observation_buffer, action))
		prediction = - self.value_old_state_action

		'''	
		Calculates TDE
		'''
		
		loss = self.backward(new_observation_buffer, prediction)

		self.finallosslist = np.append(self.finallosslist, loss)
		self.finalobservationslist = np.append(self.finalobservationslist, self.new_observation)

		return loss
	
	
	def backward(self, new_observation_buffer, prediction):
		'''
		Backward pass through backpropagation using SGD
		Calculates TD error
		Loss function is Squared TD error

		Evaluating on the mean of all losses or the loss on the new observation
		'''

		delta = new_observation_buffer + prediction
		self.loss = (delta) ** 2
		loss = torch.sum(self.loss)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return self.loss[-1].item()















class RNNState(nn.Module):
	def __init__(self, input_size, hiddenstate_size, number_recurrent_layers, dropout, nonlinearity, batch_first, bias, bidirectional, num_directions, seq_len, batch, output_size):
		super(RNNState, self).__init__()

		'''
		Initialize the parameters for the RNN state representation
		'''

		self.input_size = 2*input_size+2*input_size
		self.hiddenstate_size = hiddenstate_size
		self.number_recurrent_layers = number_recurrent_layers
		self.dropout = dropout
		self.nonlinearity = nonlinearity
		self.batch_first = batch_first
		self.bias = bias
		self.bidirectional = bidirectional
		self.num_directions = num_directions
		self.seq_len = seq_len
		self.batch = batch
		self.output_size = output_size
		
		self.hiddenstate = torch.zeros(self.number_recurrent_layers * self.num_directions,
									   self.batch, self.hiddenstate_size)
		
		self.rnn = nn.RNN(input_size = self.input_size , hidden_size = self.hiddenstate_size, 
						  num_layers = self.number_recurrent_layers, bias = self.bias,
						  batch_first = self.batch_first, dropout = self.dropout,
						  nonlinearity = self.nonlinearity, bidirectional = self.bidirectional)
		
		self.fc = nn.Linear(self.hiddenstate_size, self.output_size)


	def forward(self, input):
		'''
		Forward pass through the state representation
		'''
		
		'''
		input = input.view(self.seq_len, self.input_size)
		self.prediction = torch.tensor([])
		for i in range(len(input)):
			self.output, self.hiddenstate = self.rnn(input[i].view(1, self.batch, self.input_size), self.hiddenstate)
			
			temp_pred = self.fc(self.output).view(1)
			self.prediction = torch.cat([self.prediction, temp_pred])
		
		self.hiddenstate = self.hiddenstate.detach()
		'''

		input = input.view(self.seq_len, self.batch, self.input_size)
		self.output, self.hiddenstate = self.rnn(input, self.hiddenstate)
		self.prediction = self.fc(self.output).view(self.seq_len)
		self.hiddenstate = self.hiddenstate.detach()
		
		return self.prediction