#!/usr/bin/env python
import parse_data as pd
import validator as val
import logistic_regressor as log_r 
import linear_regressor as lin_r
import naive_bayes as bayes
import numpy as np
#python script to make predictions

#Read Data recast and split into training set and labels
data = pd.read_raw('./Project1_data.csv')
recasted_data = pd.recast(data)
train_data , labels = pd.split_data(2015,9,20,1,1,recasted_data, 1)

#features = pd.create_features_classification(train_data,2015,9,20)
features = pd.create_features_regression(train_data)



#Compute Training Rate as function of the number of features used to predict.
training_rate= 0.0000000000001  / (10**(len(features[0])-1))

#Validation Error Printing

#Run Validator
tester = val.validator(features,labels,'linear',5,0.5,training_rate, 1000000 )
a = tester.validate()

average_validation_err = np.average([i[1] for i in a])
average_train_err = np.average([i[2] for i in a])

print "LINEAR REGRESSION: Validation Err: " + str(average_validation_err) + " Train Err: " + str(average_train_err)
	

#Use Model on the Test Data

lr = lin_r.linear_regressor(features,labels,training_rate, 1000000)
train_err = lr.train() 
print "Training Error for Final Test: " + str(train_err)
recasted_data = pd.recast_keepall(data)
#print len(recasted_data)
features = pd.create_features_regression(recasted_data)
labels = lr.test(features)
#Write Results in proper time format.
outfile = open('results.csv','w')
for i in labels:
	if i<0:
		i=0
	hours = int(i/3600)
	minutes = int(i/60)%60
	seconds = int(i)%60
	outfile.write('{0:02d}:{1:02d}:{2:02d}\n'.format(hours,minutes,seconds))


#Final Training Error: 
#0.480692454301


# Computing Control :error when using previous marathon time:
# participant_number = 0
# sum_err = 0.0
# count = 0
# first_run_time = 0.0
# fr_count = 0
# for participant in train_data:
# 	t_predict = pd.most_recent_marathon_time(participant)
# 	t_label = labels[participant_number]
# 	participant_number += 1

# 	if t_predict < 0:
# 		first_run_time += t_label
# 		fr_count += 1
# 		t_predict = 16944
# 		#print t_label
# 		#continue

# 	err = (t_predict - t_label)/60.0/60.0
# 	sum_err += err*err
# 	count += 1

# 	simple_average_err = sum_err/count 

# print simple_average_err
#This was found empirically to be 16944 average marathon first run in seconds
#print first_run_time / fr_count 
	




#if __name__ =="__main__":
#	main()