
import numpy as np
import linear_regressor as lin_r
import logistic_regressor as log_r
import naive_bayes as nb

#
#from sklearn import linear_model
import random
import math

class validator():
	def __init__(self,x,y, classifier = "linear", num_test = 5, threshold = 0.5,  learn_rate = 0.001,learn_time = 1000):
		self.learn_rate = learn_rate
		self.learn_time = learn_time
		self.classifier = classifier
		self.threshold = threshold
		#set the object constructor to the constructor for the proper classifier
		if classifier == "linear" : 
			self.lr = lin_r.linear_regressor
		elif classifier == "logistic":
			self.lr = log_r.logistic_regressor
		elif classifier == "bayes":
			self.lr = nb.naive_bayes
		


		

		#break up data sets	
		#randomize the order
		self.data = zip(x,y)
		self.num_datapoints = len(self.data)
		self.num_test = num_test
		if num_test > self.num_datapoints:
			self.num_test = self.num_datapoints
		self.group_size = self.num_datapoints / num_test
		
		random.shuffle(self.data)
		#print str(self.data)
	
	''' Find the error when trained and tested on different subsets of the data
		Return the average validation error for all groups
	'''
	def validate(self):
		#print "Running validation for "+str(self.classifier)

		error_vector = []
		test_number = 0
		for ii in range(0,self.num_datapoints,self.group_size):
			temp = []
			#Split the dataset into a validate group and a train group
			test_number += 1
			#print str(test_number)
			if test_number > self.num_test:
				break
			if(ii+self.group_size>self.num_datapoints or test_number == self.num_test):
				validate_group = self.data[ii:]
				#Ensure that the for loop will be cancelled
				ii = self.num_datapoints + 1
			else:
				validate_group = self.data[ii:ii+self.group_size:1]

			

			train_group = self.data[0:ii] + self.data[ii+self.group_size:self.num_datapoints] # something like this
			# print str([i[0] for i in train_group[:]]) # For test purposes
			#print str(validate_group)

			train_err = 0.0
			#Create and train new regressor with input data as the train_group x values y values
			#update this to return the validation_err as the percentage of labels that were right returns the percentage of false labels
			if self.classifier == "bayes":
				#print "Beginning training validation for " + str(self.classifier)
				regressor = self.lr([i[0] for i in train_group[:]],[i[1] for i in train_group[:]])
				validation_err = 0.0
				true = 0
				t_p = 0
				t_n = 0
				f_p = 0
				f_n = 0
				total = 0.0
				for i in validate_group[:] :
					total += 1.0
					result = regressor.test(i[0])
					actual = i[1]
					if result == actual and result == 1:
						t_p += 1
					elif result == actual:
						t_n += 1 
					elif result ==1:
						f_p += 1
					else:
						f_n += 1
					#validation_err += (result-actual)*(result-actual)
				train_err = regressor.train_error()
				validation_err = float(f_p + f_n)/ total
				sensitivity = float(t_p)/(t_p +f_n)
				specificity  = float(t_n)/(t_n+f_p)
				if math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n))) == 0:
					MCC = 0 
				else:
					MCC = (float(t_p)*t_n - float(f_p)*f_n)/math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n)))
				
				
				temp.append(validation_err)
				temp.append(train_err)
				temp.append(sensitivity)
				temp.append(specificity)
				temp.append(MCC)
				#temp.append(train_err)

			elif self.classifier == 'linear':
				regressor = self.lr([i[0] for i in train_group[:]],[i[1] for i in train_group[:]], self.learn_rate, self.learn_time)

				train_err=regressor.train()
				print "weights found : " + str(regressor.weights)
				#Find error when tested on the test group
				#print str([i[0] for i in validate_group[:]])
				result = regressor.test([i[0] for i in validate_group[:]])
				actual = np.array([i[1] for i in validate_group[:]])
				
				validation_err = np.dot((result - actual),(result-actual))
				#average_err =
				#average_err = np.dot(np.absolute(result-actual), [1]*len(result))/len(result)
				err_per_hour = ((result-actual)/60.0)/60.0
				#err_per_hour = result - actual
				average_err = float(np.dot(err_per_hour,err_per_hour))/len(err_per_hour)
				temp.append(validation_err)
				temp.append(average_err)
				temp.append(train_err)

			else:
				#print "Beginning training validation for " + str(self.classifier)                
				regressor = self.lr([i[0] for i in train_group[:]],[i[1] for i in train_group[:]], self.learn_rate, self.learn_time, self.threshold) 
				#logistic = linear_model.LogisticRegression()
				#logistic.fit([i[0] for i in train_group[:]],[i[1] for i in train_group[:]])
				regressor.train()
				print "weights found : " + str(regressor.weights)
				validation_err = 0.0
				true = 0
				t_p = 0
				t_n = 0
				f_p = 0
				f_n = 0
				total = 0.0
				for i in validate_group[:] :
					total += 1.0
					result = regressor.test(i[0])
					
					#result = logistic.predict(i[0])
					actual = i[1]
					if result == actual and result == 1:
						t_p += 1
					elif result == actual:
						t_n += 1 
					elif result ==1:
						f_p += 1
					else:
						f_n += 1
				validation_err = float(f_p + f_n)/ total
				train_err = regressor.train_error()
				validation_err = float(f_p + f_n)/ total
				sensitivity = float(t_p)/(float(t_p +f_n))
				specificity  = float(t_n)/(float(t_n+f_p)) 
				if math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n))) == 0:
					MCC = 0 
				else:
					MCC = (float(t_p)*t_n - float(f_p)*f_n)/math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n)))
				temp.append(validation_err)
				temp.append(train_err)
				temp.append(sensitivity)
				temp.append(specificity)
				temp.append(MCC)
				#temp.append(train_err)
			
			error_vector.append(temp)

		#return np.average(error_vector)
		return error_vector

if __name__ =="__main__":
	data = [[6.0,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5.0,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9],[5.2,125,8]]
	labels = [1,1,1,1,0,0,0,0,0]
	#ii = 1.0
	#while ii > 0.0001:
	v = validator(data,labels,"logistic",4)
	print str(v.validate())
	#	print str(ii) + "Validation Error: " + str(v.validate())
	#	ii = ii/10


