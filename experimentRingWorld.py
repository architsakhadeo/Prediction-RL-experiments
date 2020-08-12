from Environments.RingWorldEnvironment import RingWorld
from Agents.RingWorldAgents.agentRNNExpectedSarsa import RNNAgent

import time
import sys

seed = int(sys.argv[1])
runs = [(seed+1+(i*10)) for i in range(1)]
timesteps = 100000



for run in runs:

	end = False
	environment = RingWorld()
	agent = RNNAgent(gamma = 0.0, truncation_length = 10, learning_rate = 3.007286598217175e-05)
	observation = environment.start(run)
	action = agent.start(observation, run)

	for i in range(timesteps):
		if i == timesteps - 1:
			end = True
		observation, reward = environment.step(action)
		action = agent.step(observation, reward, end, run+1)
