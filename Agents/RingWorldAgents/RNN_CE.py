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
from sklearn.metrics import classification_report
#torch.autograd.set_detect_anomaly(True)

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

		self.output_size = 2
		self.hiddenstate = torch.zeros(self.number_recurrent_layers * self.num_directions, self.batch,
										self.hiddenstate_size)
		
		self.rnn = nn.RNN(input_size = self.input_size , hidden_size = self.hiddenstate_size, 
		                	num_layers = self.number_recurrent_layers, bias = self.bias,
							batch_first = self.batch_first, dropout = self.dropout,
							nonlinearity = self.nonlinearity, bidirectional = self.bidirectional)
		
		self.fc = nn.Linear(self.hiddenstate_size, self.output_size)


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
		self.prediction = self.output.view(1,self.output_size)
		return self.prediction


class RNNAgent(BaseAgent):
	def __init__(self, gamma):
		self.gamma = gamma

	def start(self, observation):
		self.total_loss = 0
		self.flag = False
		self.observations_list = []
		self.predictions_list = []
		self.loss_list = []
		self.buffer_length = 1000

		self.model = RNNState()
		self.timesteps = 0
		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

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
			#print("Accuracy over last ", self.buffer_length ," timesteps: ", accuracy)
			print("Loss: ", loss)
			print(classification_report(self.observations_list, self.predictions_list))
			print('-------------------------------')
			self.flag = False
		self.action = torch.randint(0, 2, (1,))[0]
		self.old_observation = self.new_observation
		return self.action
	
	
	def train(self, old_observation, new_observation, action, reward):
		prediction = self.model.forward(old_observation, action)
		loss = self.backward(new_observation, prediction, reward)
		
		self.observations_list.append(new_observation.item())
		self.predictions_list.append(torch.max(prediction.view(2),0)[1].item())
		self.loss_list.append(loss)
		
		if len(self.loss_list) > self.buffer_length:
			self.observations_list = self.observations_list[-self.buffer_length:]
			self.predictions_list = self.predictions_list[-self.buffer_length:]
			self.loss_list = self.loss_list[-self.buffer_length:]

		return loss
	
	
	def backward(self, new_observation, prediction, reward):
		new_observation = new_observation.long()
		#self.loss = self.criterion(prediction, new_observation)
		self.loss = self.criterion(prediction, torch.tensor([reward]).long())
		loss = self.loss.item()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		return loss