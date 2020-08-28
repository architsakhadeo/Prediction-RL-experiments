import torch
import torch.nn as nn
import numpy as np
import pickle
import os

'''
old variables like old_action are variables at time t-1 that are already taken
old_action is the action the agent takes at old_observation (t-1) to get new_observation (t)
new variables like new_action are variables at time t that are yet to be taken
'''

class RNNAgent():
	def __init__(self, gamma, truncation_length, learning_rate, hiddenstate_size, ringworld_size):
		self.gamma = gamma
		self.truncation_length = truncation_length
		self.learning_rate = learning_rate
		self.hiddenstate_size = hiddenstate_size
		self.ringworld_size = ringworld_size

	def start(self, old_observation, run, old_fullstate):
		'''
		Agent starts with some observation and returns an action. Initializes parameters for the agent.
		Sets random seed
		'''

		np.random.seed(run)
		torch.manual_seed(run)

		self.timesteps = 0

		self.finallosslist = np.array([])
		self.finalobservationslist = np.array([])

		self.old_observation_buffer = torch.tensor([], dtype=torch.float)
		self.new_observation_buffer = torch.tensor([], dtype=torch.float)
		self.old_action_buffer = torch.tensor([], dtype=torch.int)
		self.new_action_buffer = torch.tensor([], dtype=torch.int)
		self.rho_buffer = torch.tensor([], dtype=torch.float)
		self.fullstate = old_fullstate

		'''
		Initializes the state representation, loss function for the agent, and the optimizer for the agent.
		'''

		self.model = RNNState(input_size=1, hiddenstate_size=self.hiddenstate_size, number_recurrent_layers=1,
		                    dropout=0.0, nonlinearity='tanh', batch_first=False, bias=True,
							bidirectional=False, num_directions=1, seq_len=self.truncation_length, batch=1, output_size=1)

		self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

		'''
		Agent returns an action. New observation becomes old observation as agent steps into time
		'''		

		self.old_action = torch.randint(0, 2, (1,))[0].int()

		return self.old_action


	def step(self, new_observation, reward, end, run, new_fullstate):
		'''
		Agent receives an observation and a reward for the action it took earlier
		It returns another action based on this observation and reward
		'''

		self.timesteps += 1

		if self.timesteps % 10000 == 0:
			print(self.timesteps)

		self.new_observation = torch.tensor(new_observation, dtype=torch.float).view(1)
		self.new_observation_buffer = torch.cat([self.new_observation_buffer, self.new_observation])[-self.truncation_length:]

		if len(self.new_action_buffer) == self.truncation_length:
			'''
			Agent learns by forward pass and backward pass through backpropagation
			and returns the loss
			'''			
			loss = self.train(self.old_observation_buffer, self.new_observation_buffer, self.old_action_buffer, self.new_action_buffer, self.fullstate)
			'''
			Agent returns an action		
			New observation becomes old observation as agent steps into time
			'''

		self.old_observation_buffer = self.new_observation_buffer
		self.old_action_buffer = torch.cat([self.old_action_buffer, self.old_action.view(1)])[-self.truncation_length:]

		self.fullstate = new_fullstate

		self.new_action = torch.randint(0, 2, (1,))[0].int()
		self.new_action_buffer = torch.cat([self.new_action_buffer, self.new_action.view(1)])[-self.truncation_length:]

		
		if self.new_action == torch.tensor(1):
			self.rho_buffer = torch.cat([self.rho_buffer, torch.tensor(0.0).view(1)])[-self.truncation_length:]
		elif self.new_action == torch.tensor(0):
			self.rho_buffer = torch.cat([self.rho_buffer, torch.tensor(2.0).view(1)])[-self.truncation_length:]

		self.old_action = self.new_action

		if end == True:
			dirpath = 'testData/RingWorldtestData/' + 'ringsize=' + str(self.ringworld_size) + '_' + 'gamma=' + str(self.gamma) + '_' + 'hiddenunits=' + str(self.hiddenstate_size)  + '_' + 'trunc=' + str(self.truncation_length) + '_' + 'learnrate=' + str(self.learning_rate) + '/'
			if not os.path.exists(dirpath):
				os.makedirs(dirpath)
			pickle.dump(self.finallosslist, open(dirpath + 'run'+str(run)+'_lossSimple','wb'))
			pickle.dump(self.finalobservationslist, open(dirpath+'run'+str(run)+'_observationsSimple','wb'))
			print(dirpath)

		return self.new_action
	
	'''
	def getInput(self, observation_buffer, action_buffer):
		
		#Transform the input to the state representation using observation and action
		

		input = torch.tensor([], dtype=torch.float)
		for i in range(len(observation_buffer)):
			temp = torch.tensor([])
			
			observation = observation_buffer[i].view(1)
			action = action_buffer[i].view(1)

			if observation == torch.tensor([0]):
				temp = torch.cat([temp, torch.tensor([1,0])])
			elif observation == torch.tensor([1]):
				temp = torch.cat([temp, torch.tensor([0,1])])
			
			if action == torch.tensor([0]):
				temp = torch.cat([temp, torch.tensor([0, 0])])
			elif action == torch.tensor([1]):
				temp = torch.cat([torch.tensor([0, 0]), temp])

			input = torch.cat([input, temp])

		return input
	'''

	def getInput1(self, observation_buffer, action_buffer):
		'''
		Transform the input to the state representation using observation and action
		'''

		input = torch.tensor([], dtype=torch.float)
		for i in range(len(observation_buffer)):
			temp = torch.tensor([])
			
			observation = observation_buffer[i].view(1)
			action = action_buffer[i].view(1)

			temp = torch.cat([temp, observation])
			temp = torch.cat([temp, torch.tensor(1.0).view(1) - observation])
			if action == torch.tensor([0]):	
				temp = torch.cat([temp, torch.tensor(1.0).view(1)])
				temp = torch.cat([temp, torch.tensor(0.0).view(1)])
			elif action == torch.tensor([1]):
				temp = torch.cat([temp, torch.tensor(0.0).view(1)])
				temp = torch.cat([temp, torch.tensor(1.0).view(1)])

			input = torch.cat([input, temp])

		return input



	def train(self, old_observation_buffer, new_observation_buffer, old_action_buffer, new_action_buffer, old_fullstate):
		'''
		TD(0) update
		Forward pass to calculate prediction of value of the old state-action
		Backward pass for n-step BPTT
		Calculates loss as TD error with target as sum of cumulant
		and discounted expected value of next state-actions 
		'''
		truevalues = [torch.tensor(self.gamma**5), torch.tensor(1.0), torch.tensor(self.gamma**1), torch.tensor(self.gamma**2), torch.tensor(self.gamma**3), torch.tensor(self.gamma**4)]
		
		self.value_old_state = self.model.forward(self.getInput1(old_observation_buffer, old_action_buffer), type='old')
		
		prediction = self.value_old_state

		
		with torch.no_grad():
			self.value_new_state = self.model.forward(self.getInput1(new_observation_buffer, new_action_buffer),type='new')
			if new_observation_buffer[-1] == torch.tensor(1.0):
				gamma_v = 0
			else:
				gamma_v = self.gamma 	
			target = self.new_observation_buffer + gamma_v * self.value_new_state
		
		delta = target - prediction

		self.loss = self.rho_buffer * (delta**2)
		loss = self.loss[-1]

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		storedloss = ((( truevalues[old_fullstate] - prediction[-1])**2)**0.5).item()

		'''
		if old_fullstate == 0:	
			storedloss = ((( torch.tensor(0.0) - prediction[-1])**2)**0.5).item()
		elif old_fullstate	storedloss = ((( torch.tensor(self.gamma * 1.0) - prediction[-1])**2)**0.5).item()
		else:
		'''
	
		self.finallosslist = np.append(self.finallosslist, storedloss)
		self.finalobservationslist = np.append(self.finalobservationslist, self.new_observation)

		return storedloss
	




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


	def forward(self, input, type):
		'''
		Forward pass through the state representation
		'''
		if type == 'old':
			input = input.view(self.seq_len, self.batch, self.input_size)		
			self.hiddenstate = self.hiddenstate.detach()
			self.output, self.hiddenstate = self.rnn(input, self.hiddenstate)
			self.prediction = self.fc(self.output).view(self.seq_len)
			self.hiddenstate = self.output[0].view(1, self.batch, self.hiddenstate_size)
		elif type == 'new':
			input = input.view(self.seq_len, self.batch, self.input_size)		
			self.output, self.hiddenstatenew = self.rnn(input, self.hiddenstate)
			self.prediction = self.fc(self.output).view(self.seq_len)
		return self.prediction