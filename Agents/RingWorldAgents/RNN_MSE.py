# How to do backpropagation through time? What is truncated BPTT?

from Agents.BaseAgent import BaseAgent
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import math
import torch
import torch.nn as nn

class RNNState(nn.Module):
	def __init__(self):
		super(RNNState, self).__init__()

		self.input_size = 2+2
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

		self.output_size = 1
		self.hiddenstate = torch.zeros(self.number_recurrent_layers * self.num_directions, self.batch,
										self.hiddenstate_size)
		
		self.rnn = nn.RNN(input_size = self.input_size , hidden_size = self.hiddenstate_size, 
							num_layers = self.number_recurrent_layers, bias = self.bias,
							batch_first = self.batch_first, dropout = self.dropout,
							nonlinearity = self.nonlinearity, bidirectional = self.bidirectional)
		
		self.fc = nn.Linear(self.hiddenstate_size, self.output_size)
		#self.relu = nn.ReLU() # predictions are positive, but using relu does not allow for negative TDE errors and leads to no learning

	def forward(self, old_observation, action):
		if old_observation == torch.tensor([0]):
			old_observation = torch.tensor([1,0], dtype=torch.float).view(2)
		elif old_observation == torch.tensor([1]):
			old_observation = torch.tensor([0,1], dtype=torch.float).view(2)
		if action == 0:
			input = torch.cat((old_observation, torch.tensor([0, 0], dtype=torch.float)))
		if action == 1:
			input = torch.cat((torch.tensor([0, 0], dtype=torch.float), old_observation))
		
		
		input = input.view(self.seq_len, self.batch, self.input_size)
		self.output, self.hiddenstate = self.rnn(input, self.hiddenstate)
		self.hiddenstate = self.hiddenstate.detach()
		self.output = self.output.view(self.hiddenstate_size)
		self.output = self.fc(self.output)
		#self.output = self.relu(self.output) 
		self.prediction = self.output[0]
		return self.prediction


class RNNAgent(BaseAgent):
	def __init__(self, gamma):
		self.gamma = gamma

	def start(self, observation):
		self.total_loss = 0
		self.flag = False
		self.loss_list = []
		self.buffer_length = 1000

		self.model = RNNState()
		self.timesteps = 0
		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000001)

		return self.action


	def step(self, observation, reward):
		self.timesteps += 1
		if self.timesteps % 1000 == 0:
			self.flag = True

		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		loss = self.train(self.old_observation, self.new_observation, self.action, reward)

		if self.flag == True:
			loss = sum(self.loss_list)*1.0/len(self.loss_list)
			print("Timestep: ", self.timesteps)
			print("Loss: ", loss)
			print('-------------------------------')
			self.flag = False
		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation
		return self.action
	
	
	def train(self, old_observation, new_observation, action, reward):
		self.value_oldstate = self.model.forward(old_observation, action)
		value_newstate_0 = self.model.forward(new_observation, torch.tensor(0))
		value_newstate_1 = self.model.forward(new_observation, torch.tensor(1))
		if value_newstate_0 > value_newstate_1:
			self.max_value_newstate = value_newstate_0
		else:
			self.max_value_newstate = value_newstate_1
		
		#MSE
		prediction = - self.value_oldstate

		loss = self.backward(new_observation, prediction, reward)
		#if new_observation[0] == 1:
		#	print("Observation: ", new_observation[0], " Value: ", self.value_oldstate, "	 Loss: ", loss)
		#if reward == 1:
		print("Reward: ", reward, " Value: ", self.value_oldstate, "	 Loss: ", loss)
		
		self.loss_list.append(loss)
		
		if len(self.loss_list) > self.buffer_length:
			self.loss_list = self.loss_list[-self.buffer_length:]

		return loss
	
	
	def backward(self, new_observation, prediction, reward):
		new_observation = new_observation.long()
		#delta = new_observation[0] + prediction
		delta = reward + prediction
		self.loss = (delta) ** 2
		loss = self.loss.item()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		return loss