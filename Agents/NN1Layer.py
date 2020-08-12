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

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.input_size = 1
		self.output_size = 2
		#self.fc = nn.Linear(self.input_size, self.output_size)
		self.fc1 = nn.Linear(self.input_size, 1024)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(1024, self.output_size)
		#self.softmax = nn.Softmax()

	def forward(self, old_observation):
		#self.output = self.fc(old_observation)
		#self.softmax_output = self.softmax(self.output)
		
		self.output = self.fc1(old_observation)
		self.output = self.tanh(self.output)
		self.output = self.fc2(self.output)
		self.prediction = self.output.view(1,self.output_size)
		return self.prediction


class Linear(BaseAgent):
	def __init__(self, gamma):
		self.gamma = gamma

	def start(self, observation):
		self.total_loss = 0
		self.model = Model()
		self.timesteps = 0
		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		self.action = 0
		self.old_observation = self.new_observation
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
		return self.action


	def step(self, observation):
		self.timesteps += 1
		self.new_observation = torch.tensor(observation, dtype=torch.float).view(1)
		loss = self.train(self.old_observation, self.new_observation)
		self.total_loss += loss
		print(self.total_loss*1.0/self.timesteps)
		#for param in self.model.parameters():
		#	print(param.data)
		self.action = 0
		self.old_observation = self.new_observation
		return self.action
	
	
	def train(self, old_observation, new_observation):
		prediction = self.model.forward(old_observation)
		loss = self.backward(new_observation, prediction)
		return loss
	
	
	def backward(self, new_observation, prediction):
		self.optimizer.zero_grad()
		
		print(new_observation, ' | ', prediction, end = ' | ')
		new_observation = new_observation.long()
		self.loss = self.criterion(prediction, new_observation)
		loss = self.loss.item()
		self.loss.backward()
		self.optimizer.step()

		return loss