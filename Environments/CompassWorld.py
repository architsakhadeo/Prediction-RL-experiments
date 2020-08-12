from Environments.BaseEnvironment import BaseEnvironment
import numpy as np
import math

'''
---> full state space = array([colour in front, x-coordinate, y-coordinate, absolute orientation])
	---> example of full state = array(['white', 2, 3, 'north'])

---> observations = [one hot binary vector of colors]
	---> example of observations = [0, 1, 0, 0, 0, 0]

--> action space = array(['left', 'forward', 'right'])
'''

class CompassWorld(BaseEnvironment):
	def __init__(self):
		self.state = None
		self.pseudostate = None
		self.observations = None

		self.colours = ['white', 'red', 'yellow', 'orange', 'green', 'blue']
		self.orientation = ['north', 'east', 'south', 'west']
		self.actions = ['left', 'forward', 'right']

		self.xy_lowerbound = [0, 0]
		self.xy_upperbound = [5, 5]
		
		self.x_length = self.xy_upperbound[0] - self.xy_lowerbound[0] + 1
		self.y_length = self.xy_upperbound[1] - self.xy_lowerbound[1] + 1
		
	def partialobservability(self, state):
		return state[0:6]
	
	def constructstate(self, pseudostate):
		'''
		Usage

        #self.state = self.constructstate(self.pseudostate)
		#self.observations = self.partialobservability(self.state)
        '''
		
		# 6 + 6*6*4
		# 6 bits for colour
		# 36 bits for position
		# 4 bits for orientation

		clr, x, y, ornt = pseudostate
		
		clrstate = np.zeros(len(self.colours))
		position_state = np.zeros(self.x_length * self.y_length)
		position_orntstate = np.zeros(self.x_length * self.y_length * len(self.orientation)) 

		clrstate[self.colours.index(clr)] = 1
		position_state[self.y_length * y + x] = 1
		position_orntstate[self.orientation.index(ornt)*self.x_length*self.y_length: (self.orientation.index(ornt)+1)*self.x_length*self.y_length] = position_state
		
		state = np.concatenate((clrstate, position_orntstate))

		return state
		
	def pseudo2observations(self, pseudostate):
		clr, x, y, ornt = pseudostate
		clrstate = np.zeros(len(self.colours))
		clrstate[self.colours.index(clr)] = 1

		return clrstate

	def start(self, run):
		np.random.seed(run)
		
		self.steps_beyond_done = None
		self.pseudostate = np.array(['white', 0, 0, 'east'], dtype=object)
		self.observations = self.pseudo2observations(self.pseudostate)

		return self.observations

	def step(self, action):
		colour, xcoord, ycoord, orient = self.pseudostate

		# Decide x and y coordinates, other coordinates and orientation remains same
		if action == [0,1,0]:
			if orient == 'north':
				if ycoord < self.xy_upperbound[1]:
					ycoord += 1
			elif orient == 'south':
				if ycoord > self.xy_lowerbound[1]:
					ycoord -= 1
			elif orient == 'east':
				if xcoord < self.xy_upperbound[0]:
					xcoord += 1
			elif orient == 'west':
				if xcoord > self.xy_lowerbound[0]:
					xcoord -= 1
						
		# coordinates remain same, orientation changes
		elif action == [1,0,0]:
			if orient == 'north':
				orient = 'west'
			elif orient == 'south':
				orient = 'east'
			elif orient == 'east':
				orient = 'north'
			elif orient == 'west':
				orient = 'south'
		
		elif action == [0,0,1]:
			if orient == 'north':
				orient = 'east'
			elif orient == 'south':
				orient = 'west'
			elif orient == 'east':
				orient = 'south'
			elif orient == 'west':
				orient = 'north'

		# Decide wall colour
		if ycoord == self.xy_lowerbound[1] and orient == 'south':
			colour = 'red'
		elif ycoord == self.xy_upperbound[1] and orient == 'north':
			colour = 'orange'
		elif xcoord == self.xy_upperbound[0] and orient == 'east':
			colour = 'yellow'
		elif xcoord == self.xy_lowerbound[0] and orient == 'west':
			if ycoord == self.xy_upperbound[1]:
				colour = 'green'
			else:
				colour = 'blue'
		else:
			colour = 'white'
		
		self.pseudostate = np.array([colour, xcoord, ycoord, orient], dtype=object)
		self.observations = self.pseudo2observations(self.pseudostate)

		if colour != 'white':
			term = 0
		else:
			term = 1

		return self.observations, term
			
