class BaseExperiment:
	def __init__(self, agent, env):
		self.environment = env
		self.agent = agent

		self.last_action = None
		self.total_reward = 0.0
		self.num_steps = 0
		self.num_episodes = 0	

	def start(self):
		self.num_steps = 0
		self.total_reward = 0

		s = self.environment.start()
		obs = self.observationChannel(s)
		self.last_action = self.agent.start(obs)

		return (obs, self.last_action)

	def step(self):
		(s, reward, term) = self.environment.step(self.last_action)
		obs = self.observationChannel(s)

		self.total_reward += reward

		self.num_steps += 1
		self.last_action = self.agent.step(reward, obs, term)
		
		if term:
			self.num_episodes += 1

		return term

	def runEpisode(self, max_steps = 0):
		is_terminal = False
		self.start()
		while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
			is_terminal = self.step()

		return is_terminal	
		
	def observationChannel(self, s):
		return s
