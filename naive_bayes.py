import numpy as np

class naive_bayes():

	'''take in the data points and find all the underlying distributions'''
	def __init__ (self, x, y): 
		self.data = np.array(x)
		self.dim = len(self.data[0])
		self.labels = np.array(y)
		self.y1 = self.find_prior(self.labels)
		self.y0 = 1.0 - self.y1
		
		self.distribution_table = []	#Distribution table stores the mean and variance for every feature given a specific label
										#Variance is stored as -1 if the label is discrete.


		for jj in range(0,self.dim):
			attribute = [x[jj] for x in self.data[:]]
			#print str(attribute)
			mean_1, var_1 = self.find_gaussian_likelihood(attribute, self.labels, 1)
			mean_0, var_0 = self.find_gaussian_likelihood(attribute, self.labels, 0)

			self.distribution_table.append([[mean_0,var_0],[mean_1,var_1]])

	''' return the value of y that best fits the features given, where the likelihood is the product of the posterior and the likelihood of individual attributes given each label'''
	def test(self,x):

		log_likelihood_ratio =  np.log(self.y1/self.y0)
		for ii in range(0,self.dim):

			#If the feature is continuous then add the log ratio of the probabilities according to the learned gaussian distribution
			#we use the variance as the flag for whether or not the variable is continuous or not. negative variance = binary group 
			if self.distribution_table[ii][1][1] <0 and self.distribution_table[ii][0][1] <0:
				# if x[ii] = 1 then the mean for both negative and positive labels will be the corresponding probability : sum of x =1 / y=0 or sum of x=1 /y=1 
				if x[ii] == 1:
					denom = self.distribution_table[ii][0][0]
					if denom <= 0:
						log_likelihood_ratio += 1

					else:
						log_likelihood_ratio = log_likelihood_ratio + np.log(self.distribution_table[ii][1][0]/ self.distribution_table[ii][0][0])
				#x[ii] = 0 = (1-  p(x_ii = 1 | y))
				else : 
					denom = (1-self.distribution_table[ii][0][0])
					if denom<=0:
						log_likelihood_ratio+=1

					else:
						log_likelihood_ratio = log_likelihood_ratio  + np.log((1-self.distribution_table[ii][1][0])/(1-self.distribution_table[ii][0][0]))
			#continuous case
			else:
				denom = self.gaussian_prob(self.distribution_table[ii][0][0], self.distribution_table[ii][0][1], x[ii])
				if denom <= 0.00000001:
					log_likelihood_ratio += 1
					#print ii
					#print 'c'
					#print x[ii]
				else:
					log_likelihood_ratio = log_likelihood_ratio + np.log(self.gaussian_prob(self.distribution_table[ii][1][0], self.distribution_table[ii][1][1], x[ii])/  denom)
			# Otherwise usethe probability according to a binomial dist

		#print "0: " + str(likelihood_0) + " 1: " + str(likelihood_1)
		if log_likelihood_ratio <0:
			return 0
		else :
			return 1
		

	'''Find p(x_i|y=class) = gaussian, where x_i cooresponds to one attribute , returns the mean and variance
	Assumes that binary features are stored as 0 and 1
	Returns mean = mean, var = -1 if the feature is binary.
	'''
	def find_gaussian_likelihood(self,x_i,y,label):
		mean = 0.0
		count = 0
		continous_flag = 0 # if the x[ii] are found to be continous
		for ii in range(0,len(y)):
			if not (x_i[ii] == 1 or x_i[ii] == 0) : 
				continous_flag = 1 # update the flag if a non binary value is found
			
			if y[ii] == label: 
				mean = mean + x_i[ii]
				count = count+1
		if count == 0:
			mean = 0.0
		else:
			mean = mean / count
		#if the variable is infact discrete, then we will return a negative variance as a flag that the variable is binary.	
		if continous_flag == 0:
			return mean, -1

		var = 0.0
		for ii in range(0,len(y)):
			if y[ii] == label: 
				var = var + (float(x_i[ii]) - mean)*(float(x_i[ii]) - mean)
		
		if count ==1:
			var = 0
		else:
			var = var / (count-1)

		return mean, var

	'''Returns the likelihood of x according to a gaussian distribution given the mean and var for a specific label and attribute'''
	def gaussian_prob(self,mean,var, x):
		if var ==0:
			var = 0.01
		return (1/np.sqrt(2*3.141592653585*var))*(np.exp(-((x-mean)*(x-mean))/(2*var)))

	''' returns the probability that y =1 from the label set '''
	def find_prior(self,y): 
		occurences = 0.0
		for ii in range(0,len(y)):
			if y[ii] ==1 :
				occurences = occurences + 1.0
		#print str(occurences) + "   " + str(len(y))
		return occurences / float(len(y))

	''' compute the number of wrong labels returned/total , as well as the sensitivity and specificity when using data from the training set'''
	def train_error(self):
		t_p = 0
		t_n = 0
		f_p = 0
		f_n = 0
		for ii in range(0,len(self.labels)):
			predicted = self.test(self.data[ii])
			actual = self.labels[ii]
			#determine true/false positive or negative labels . True = match, positive = 1
			if actual == 1:
				if predicted == 1:
					t_p +=1
				else : 
					f_n += 1
			else:
				if predicted == 1:
					f_p +=1
				else:
					t_n += 1

		return float(f_p + f_n)/len(self.labels) , t_p/float(t_p+ f_n) , t_n/float(t_n + f_p)


if __name__ == "__main__":
	#Test Data taken from Wikipedia inorder to test our results against theirs
	x = [[6.0,180,1],[5.92,190,1],[5.58,170,1],[5.92,165,1],[5.0,100,0],[5.5,150,0],[5.42,130,0],[5.75,150,0]]
	y = [1,1,1,1,0,0,0,0]
	test_x = [6.0,160,0]
	nb = naive_bayes(x,y)
	print str(nb.train_error())
	print str(nb.test(test_x))
