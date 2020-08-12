from Environments.CompassWorld import CompassWorld
from Agents.CompassWorldAgents.RNN_TDE import RNNAgent

'''
Create instances of the environment and agent
'''

'''
Starts the environment and the agent
The environment generates the first state and the agent takes the first action
'''


'''
A continuing task that undergoes the following number of iteration steps.
'''

import time
import sys

runs = int(sys.argv[1])
timesteps = 100000
for run in range(runs, runs+1):
	end = False
	environment = CompassWorld()
	agent = RNNAgent(gamma=1.0)
	observation = environment.start(run)
	action = agent.start(observation, run)

	for i in range(timesteps):
		if i == timesteps - 1:
			end = True
		observation, term = environment.step(action)
		action = agent.step(observation, term, end, run+1)