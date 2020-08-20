import numpy as np

'''
MDP =            -> 1 <-> 0 <-> 0 <-> ..... <-> 0 <-
                |                                   |
				 -----------------------------------						

Full state = k , 0 <= k <= n
Partial state = {0, 1}  
Reward = {0, 1}
Reward = Partial state
'''

class RingWorld():
	def __init__(self, ringworld_size):
		'''
		Initialize parameters of the environment
		'''

		#Long rings and just 1 observation is very difficult

		self.fullstate = None
		self.partialstate = None
		self.reward = 0
		self.ringsize = ringworld_size #100 and 1 vs 1000 and 9:1
		self.lowerbound = 0
		self.upperbound = self.ringsize-1
		self.random_states_reward1 = [0]
		#self.random_states_reward1 = np.random.choice(self.ringsize, int(self.ringsize/10), replace=False)
		
	def partialobservability(self, fullstate):
		'''
		Convert full state to a partially observable state
		'''

		if fullstate == 0:
			partialstate = 1
		else:
			partialstate = 0
		return partialstate
			

	def start(self, run):
		'''
		Environment starts with a random state

		Can be random, but the problem is learning starts only after encountering the first state
		with observation and reward = 1
			self.fullstate = np.random.choice(self.ringsize)
		'''
		np.random.seed(run)

		self.timesteps = 0
		self.fullstate = 0
		self.partialstate = self.partialobservability(self.fullstate)
		return self.partialstate, self.fullstate

	def step(self, action):
		'''
		Environment steps by taking the agent's action and returns
		the next state and reward
		'''
		self.timesteps += 1

		if action == 0:
			if self.fullstate == self.lowerbound:
				self.fullstate = self.upperbound
			else:
				self.fullstate -= 1
		elif action == 1:
			if self.fullstate == self.upperbound:
				self.fullstate = self.lowerbound
			else:
				self.fullstate += 1
		
		self.partialstate = self.partialobservability(self.fullstate)

		if self.fullstate in self.random_states_reward1:
		#if self.partialstate == 1:
			self.reward = 1
		else:
			self.reward = 0


		return self.partialstate, self.reward, self.fullstate
			
