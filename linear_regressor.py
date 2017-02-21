#!/usr/bin/env python
import sys
import time
import numpy as np

class linear_regressor():
	'''
		Linear Regressor Object
		Parameters: 	x  	training data
						y 	training labels
						learning rate 	scalar value used when performing gradient descent to train. Controls the size of each update.
						learn time 		number of iterations gradient descent will run for
		
		Attributes:  	labels 		training labels(i.e. completion times)
						data		training features including a bias term for each point
						weights 	the weight of each feature in predicting final label (i.e completion time)

	'''
	def __init__(self, x,y, learning_rate = 0.00000001, learn_time = 100000):
		new_y = y[:]
		self.labels = np.array(new_y)
		new_x = []
		#deep copy x before appending the bias
		for i in x:
			new_feature = []
			for j in i:
				new_feature.append(j)
			new_x.append(new_feature)
		#add a feature that will coorespond to the bias 
		for i in new_x :
			i.append(1)

		self.data = np.array(new_x)


		#Assuming that the features are stored in the second dimension of the data, first dimension coorepsonds to the patient
		n, dim = np.shape(self.data)
		#print dim
		#initialize so the weight of every variable is even
		self.weights = [1] * dim
		self.weights[-1] = 0#3600
		#self.bias = 1 
		self.learning_rate = learning_rate
		self.learn_time = learn_time

	'''
	Trains the linear regressor using gradient descent. Number of iterations and learning rate are defined by the object's attributes.
	Returns: the mean squared error in hours.
	''' 
	def train(self):
		for ii in range(0,self.learn_time):
			weight_vector = np.array(self.weights)
			#rint weight_vector
			# Compute Derivative
			t1 = np.dot(np.dot(self.data.transpose(),self.data).transpose(),weight_vector)
			t2 =  np.dot(self.data.transpose(),self.labels)
			der = t1-t2
			
			#Update Weights
			for i in range(0, len(self.weights)) :
				self.weights[i] -= 2*der[i] *self.learning_rate
			
		
		err = self.labels - np.dot(self.data,self.weights)
		err = err /3600.0 #Convert Error from seconds to hours.

		return float(np.dot(err,err))/len(err)

	'''Using the trained linear regressor predict the completion time of each input point.
		Param: x 	vector of input data
		Returns: y 	vector of completion times cooresponding to each input point
	'''
	def test(self,x):
		new_x = []
		for i in x:
			new_feature = []
			for j in i:
				new_feature.append(j)
			new_x.append(new_feature)
		#append a bias term to the feature set.
		for i in new_x:
			i.append(1)
		y =np.dot(np.array(new_x),self.weights)
		return y
	
''' Small Test '''
if __name__ =="__main__":
	#for testing purposes
		x = [[1],[2],[3],[4]] 
		y = [6,10,14,18]
		test_x = [[5],[6]]
		l = linear_regressor(x,y,0.0001)
		l.train()

		print x
		print l.data
		print l.weights
		print str(l.test(test_x))