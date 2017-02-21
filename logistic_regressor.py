import numpy as np
import sys
import warnings


class logistic_regressor():
    '''
        Logistic Regressor Object
        Parameters:     x   training data
                        y   training labels
                        learning rate   scalar value used when performing gradient descent to train. Controls the size of each update.
                        learn time      number of iterations gradient descent will run for
                        threshold       threshold used to predict a positive or negative label
        
        Attributes:     labels      training labels(i.e. completion times)
                        data        training features
                        weights     the weight of each feature in predicting final label (i.e completion time)
                        bias        bias term used to in prediction

    '''
    def __init__(self,x,y, learn_rate = 0.001 ,learn_time = 1000, threshold = 0.5):


        self.data = np.array(x)
        self.labels = np.array(y)
        self.learn_rate = learn_rate
        self.learn_time = learn_time
        self.threshold = threshold

        self.dim = np.shape(self.data)[1]


        self.weights = [1] * self.dim
        self.bias = 1

    ''' Train the Logistic Regressor by running gradient descent for learn time iterations.'''

    def train(self):
        for ii in range(0,self.learn_time):
            if ii%200 == 0: print "Iteration of GD = " + str(ii)
            #der = self.weights
            weight_der = [0] * self.dim
            bias_der = 0
            #Add der for each test point 
            for jj in range(0,len(self.labels)):
                update_factor = (self.labels[jj]-self.sigmoid_function(self.data[jj],self.weights,self.bias))
                weight_der = weight_der + self.data[jj]* update_factor
                bias_der = bias_der + update_factor
            for ii in range(0,len(self.weights)):
                self.weights[ii] = self.weights[ii] + self.learn_rate *  weight_der[ii]
                self.bias = self.bias +self.learn_rate * bias_der

    '''Test a single point using the learned model
        Returns: predicted label of point'''
    def test(self, x):
        #append bias term
        #if train_error == 0:
         #   x.append(1)

        #print "Prob that data point is positive = " + str(self.sigmoid_function(x,self.weights))
        if self.sigmoid_function(x,self.weights,self.bias)> self.threshold : 
            return 1
        else :
            return 0

    ''' Compute the number of wrong labels returned/total , as well as the sensitivity and specificity when using data from the training set
    Returns     error, sensitivity, specifity
    '''
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

    ''' Returns the sigmoid function given parameters
        input   x       features values (np.array)
                w       feature weights (np.array) 
                bias    bias term
                '''
    def sigmoid_function(self,x,w,bias):
        np.seterr(over = 'ignore')

        w = np.array(w)
        #print str(w)
        x = np.array(x)
        #print str(x)
        y = np.dot(w,x.transpose()) + bias

        return 1/(1 + np.exp(-1*y))

'''Test'''
if __name__ == "__main__":
    # If called on own, then run basic test
    data_set = [[1,2],[3,4],[5,6],[7,8]]
    label_set = [0,0,1,1]
    test_data = [4,5]
    #this can serve as an example of how to implement lr in other functions
    lr = logistic_regressor(data_set,label_set)
    lr.train()
    print str(lr.weights) + " " + str(lr.bias)
    print str(lr.test(test_data))