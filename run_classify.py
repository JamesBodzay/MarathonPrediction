#!/usr/bin/env python
import parse_data as pd
import validator as val
import logistic_regressor as log_r 
import linear_regressor as lin_r
import naive_bayes as bayes
import numpy as np
import random as r
import math

#python script to make predictions

data = pd.read_raw('./Project1_data.csv')
recasted_data = pd.recast_keepall(data)
print len(recasted_data)

#raw_input()
train_data , labels = pd.split_data(2015,9,20,1,1,recasted_data, 0)
features = pd.create_features_classification(train_data,2015,9,20)
#Script to provide validation results
tester = val.validator(features,labels,'logistic')

a = tester.validate()
average_validation_err = np.average([i[0] for i in a])
average_train_err = np.average([i[1][0] for i in a])
average_sens = np.average([i[2] for i in a])
average_spec = np.average([i[3] for i in a])
average_mcc = np.average([i[4] for i in a])
print "LOGISTIC:: Validation Err: " + str(average_validation_err) + " Train Err: " + str(average_train_err) + "Sensitivity: " + str(average_sens) + "Specificity: " + str(average_spec) + "MCC:  " + str(average_mcc)


tester = val.validator(features,labels,'bayes')

average_validation_err = np.average([i[0] for i in a])
average_train_err = np.average([i[1][0] for i in a])
average_sens = np.average([i[2] for i in a])
average_spec = np.average([i[3] for i in a])
average_mcc = np.average([i[4] for i in a])
print "BAYES: Validation Err: " + str(average_validation_err) + " Train Err: " + str(average_train_err) + "Sensitivity: " + str(average_sens) + "Specificity: " + str(average_spec) + "MCC:  " + str(average_mcc)
#Random Control Tests
# t_p = 0
# t_n = 0
# f_n = 0 
# f_p = 0
# for i in range (0,len(labels)):
# 	ran_label = r.randint(0,1)
# 	if ran_label == labels[i]:
# 		t_p += ran_label
# 		t_n += (1-ran_label)
# 	else:
# 		f_p += ran_label
# 		f_n += (1-ran_label)

# if math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n))) == 0:
# 	MCC = 0 
# else:
# 	MCC = (float(t_p)*t_n - float(f_p)*f_n)/math.sqrt(float((t_p+f_p)*(t_p+f_n)*(t_n+f_p)*(t_n+f_n)))

# print "Random Accuracy: " + str(float(f_p+f_n)/(t_p+f_p+f_n+t_n)) + "Sens : " + str(float(t_p)/(t_p+f_n)) + "Spec: " + str(float(t_n)/(t_n+f_p)) + " MCC: " + str(MCC)



#ACTUAL TESTING
bay = bayes.naive_bayes(features,labels)
log = log_r.logistic_regressor(features,labels,0.001,5000)
log.train()
err = log.train_error()
print err
#print bay.distribution_table
print bay.train_error()
features = pd.create_features_classification(recasted_data,2016,9,25)



#print features
outfile = open('results_classifier.csv','w')
for x in features:
	#y.append(bay.test(x))
	outfile.write(str(bay.test(x))+"\t"+str(log.test(x))+"\n")

#print "done"