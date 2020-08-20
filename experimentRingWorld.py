from Environments.RingWorldEnvironment import RingWorld
from Agents.RingWorldAgents.agentRNNExpectedSarsa import RNNAgent

import time
import sys

seed = int(sys.argv[1])
runs = [(seed+1+(i*10)) for i in range(1)]
timesteps = 100000

ringworld_size = 6
gamma = 0.0
truncation_length = 16 #1,2,4,6,10,16
learning_rate = 0.0005
hiddenstate_size = 6 #3,6,9,12


for run in runs:

	end = False
	environment = RingWorld(ringworld_size = ringworld_size)
	
	agent = RNNAgent(gamma = gamma, truncation_length = truncation_length,
					learning_rate = learning_rate, hiddenstate_size = hiddenstate_size,
					ringworld_size = ringworld_size)
	
	observation, fullstate = environment.start(run)
	action = agent.start(observation, run, fullstate)

	for i in range(timesteps):
		if i == timesteps - 1:
			end = True
		observation, reward, fullstate = environment.step(action)
		action = agent.step(observation, reward, end, run+1, fullstate)
