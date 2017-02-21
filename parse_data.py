#!/usr/bin/env python
import numpy as np
import csv
import time
import datetime
from datetime import date
import unicodedata
import random 
import re
#import statistics
from datetime import date

DEFAULT_AGE = 30
INVALID_RACE_FLAG = -1
INVALID_RACE_TIME = -1

'''Read the csv file and store it as a 2 dimensional array of entries
	Param: 	filename 	location of csv file to read raw data from
'''
def read_raw(filename):
	#print str(filename)
	 with open(filename, 'r') as fptr:
	 	csv_file = csv.reader(fptr, delimiter=',')
	 	csv_file.next()
	 	data = []
	 	for participant in csv_file:
	 		data.append(participant)
	 return data

''' 
	Recast variables like M/F to being 1 if female, 0 if male, splitting age and gender into two columns... etc.
	Change the race name to being a binary value, either was the previous montreal marathon or was not the montreal marathon
	returns each participant as an array of arrays for each participant 
	Thus the returned value is a 3d array beware

	Param:	data 	raw data read from csv file
	Returns: new_data 	data recasted to so it can be used to create features.
''' 
def recast(data):
	new_data = []
	average_age = 0
	for participant in data:
		new_participant = []
		#Each race has 5 categories, thus the number of races run is len(participant) - 1 /5 
		num_races = (len(participant)-1)/5

		for race in range(0,num_races):
			#Read the date into a d m y to be put in seperate columns.
			day,month,year = get_date(participant[race*5+1])
			#Turn location/ race name into a binary value 
			race_location = is_montreal_marathon(participant[race*5+2])
			#extract distance of race from the racetype identifier
			race_distance = racetype_to_distance(participant[race*5+3])
			if race_distance > 180:
				break

			if race_distance == INVALID_RACE_FLAG:
				break
			race_time = string_to_seconds(participant[race*5+4])	
			#race_time = string_to_hours(participant[race*5+4])
			if race_time == INVALID_RACE_TIME:
				break

			#Read the category column
			category = str(participant[race*5 + 5])
			
			age = get_age(category)
			gender = get_gender(category)
			new_participant.append([day,month,year,race_location,race_distance,race_time, age,gender])
		
		#Append the whole new participant 2d array to the data, unless the new participant no longer has any data due to scrubbing  
		if len(new_participant) > 0:
			new_data.append(new_participant)

	return new_data

'''A variation of the above code that throws out no data, this is to be used when recasting for the final test set to ensure every runner is accounted for '''
def recast_keepall(data):
	new_data = []
	average_age = 0
	for participant in data:
		new_participant = []
		#Each race has 5 categories, thus the number of races run is len(participant) - 1 /5 
		num_races = (len(participant)-1)/5

		for race in range(0,num_races):
			#Read the date into a d m y to be put in seperate columns.
			day,month,year = get_date(participant[race*5+1])
			#Turn location/ race name into a binary value 
			race_location = is_montreal_marathon(participant[race*5+2])
			#extract distance of race from the racetype identifier
			race_distance = racetype_to_distance(participant[race*5+3])
			
			if race_distance > 180:
				#basic assumpion to avoid breaking 
				race_distance = 21
				#break
			# if race_distance == INVALID_RACE_FLAG:
			# 	break
			race_time = string_to_seconds(participant[race*5+4])	
			#race_time = string_to_hours(participant[race*5+4])
			# if race_time == INVALID_RACE_TIME:
			# 	break

			#Read the category column
			category = str(participant[race*5 + 5])
			
			age = get_age(category)
			gender = get_gender(category)
			new_participant.append([day,month,year,race_location,race_distance,race_time, age,gender])
		
		#Append the whole new participant 2d array to the data, unless the new participant no longer has any data due to scrubbing  
		if len(new_participant) > 0:
			new_data.append(new_participant)

	return new_data


####
'''HELPER METHODS FOR RECAST'''
####

'''returns the timestamp converted to seconds, returns INVALID_RACE_TIME if not enough parameters found, i.e. did not finish '''
def string_to_seconds(timestamp):
	hms = timestamp.split(':')
	if len(hms)< 3 :
		return INVALID_RACE_TIME
	return int(hms[0])*60*60 + int(hms[1])*60 + int(hms[2])
def string_to_hours(timestamp):
	hms = timestamp.split(':')
	if len(hms)< 3 :
		return INVALID_RACE_TIME
	return float(hms[0]) + float(hms[1])/60.0 + float(hms[2])/(60.0 * 60.0)

def racetype_to_distance(racetype):
	lower_name = racetype.lower()
	#look for a number in the racetype : assume cooresponds to the race distance in km
	dist = re.findall("\d+",lower_name)
	# if no  distance present look for common names
	if len(dist) < 1 :
		if 'demi' in lower_name or 'half' in lower_name :
			return 21
		elif 'marathon' in lower_name:
			return 42
		else:
			return INVALID_RACE_FLAG
	else: 
		return int(dist[0])

''' returns 1 if the marathon is the montreal marathon else returns 0 '''
def is_montreal_marathon(racename):
	lower_name = racename.lower()
	#looks for oasis montreal marathon
	if 'oasis'in lower_name and 'mont' in lower_name :
		return 1
	else:
		return 0

''' Returns the day , month and year from a dd-mm-yyyy format'''			
def get_date(date):
	d_array = re.findall("\d+",date)
	if len(d_array) < 3 :
		return -1, -1 , -1 
	return int(d_array[0]), int(d_array[1]), int(d_array[2])

''' Returns the gender of the participant as 1 = female , 0 = male'''
def get_gender(category):
	if 'f' in category.lower():
		#print category
		return 1
	else:
		#print 'woo'
		return 0 

'''Returns the median age in an age category given'''
def get_age(category):
	ages = re.findall("\d+", category)
	if len(ages) == 2:
		age = (int(ages[0]) + int(ages[1]))/2
	elif len(ages) == 1:
		age = int(ages[0])
	else:
		age = DEFAULT_AGE
	return age

######

'''DATA SPLITTER'''

######	

'''Splits recasted data into a label set based on whether they raced in the specific race (specified by date parameters, a montreal boolean var, and  is marathon , potentially more), removes any races past the date of concern
Input negative value to ignore the specfic parameter, desired_label = 1  label is race time, desired_label = 0 label is the did_race_label 
Returns Data, Labels '''
def split_data(race_year, race_month,race_day, in_montreal, is_marathon, recasted_data, desired_label):
	new_data = []
	new_labels = []
	query_date = date(race_year,race_month,race_day)
	for participant in recasted_data:
		
		new_participant = [] # New data with filtered data taken out
		participant_label= 0 #race time or binary value depending on desired label set
		for race in participant:

			date_location_match = 0
			race_date = date(race[0],race[1],race[2])

			if query_date ==race_date and (in_montreal <0 or in_montreal == race[3]):			
				date_location_match = 1
			#ignore races that haven't happend before the query date
			elif query_date<=race_date:
				break;

			#If the date and location match then check to make sure the marathon parameter matches	
			if date_location_match == 1 and (is_marathon<0 or (is_marathon == 0 and not (int(race[4]) == 42) ) or (is_marathon == 1 and ((int(race[4]) == 42)))) :
			 	match = 1
			# We don't want to include any races that happened the same date as the desired_race	(i.e montreal)
			elif date_location_match ==1:
				break
			
			else :
				match = 0
			#If there's a match then set the label
			if match == 1:
				#Sets the label to be the race time
				if desired_label == 1:
					participant_label = race[5]
				else:
					participant_label = 1
			#Otherwise add the race to the data set
			else:
				new_participant.append(race)

		#After all the races have been checked if the participant has a time then add it to the dataset(with the time as the label and the other races as the dataset)
		#For the did they race? classifier we add all participants with 1 or 0 as their label
		if desired_label == 1 and participant_label > 0 and len(new_participant) >0:
			
			new_data.append(new_participant)
			new_labels.append(participant_label)

		elif desired_label == 0 and len(new_participant)>0:
			new_data.append(new_participant)
			new_labels.append(participant_label)
				
	return new_data, new_labels


#######

'''FEATURES'''

#####

'''Takes as input data that is to go into a regressor or into a classifier and turns the raw data into usable features
use this for classification

returns features = collection of features from the data
'''
def create_features_classification(data,curr_year, curr_month, curr_day):
	feature_set = []
	for p in data:
		p_features = []
		p_features.append(competed_in_montreal_marathon(p,curr_year-1))

		ave_dist = average_distance(p)
		#p_features.append(ave_dist) # Good
		#print ave_dist
		#p_features.append(variance_distance(p,ave_dist))
		#p_features.append(mode_distance(p))
		#p_features.append(num_marathons(p))
		p_features.append(number_of_montreal_marathons(p)) # Good
		#number of races
		#p_features.append(len(p))
		#age
		#p_features.append(p[0][6]) # Good
		#gender
		#p_features.append(p[0][7]) # Good 
		#p_features.append(number_of_montreal_races(p))
		#p_features.append(days_since_last_race(p,curr_year,curr_month,curr_day)-average_days_between_race(p)) # Good

		feature_set.append(p_features)

	return feature_set



'''Takes as input data that is to go into a regressor or into a classifier and turns the raw data into usable features

Use this for time prediction
returns features = collection of features from the data
'''
def create_features_regression(data):
	feature_set = []
	#In order to fill in empty features for other participants we will fill it in with average of all the other participants
	feature_sums = []
	feature_counts = []
	for p in data:
		#US average is 4:20:27 =162000 seconds, if no marathons run then use previous time as this average
		default_marathon_time = 16944
		
		p_features = []
		ave_mar_time = average_marathon_time(p)
		if ave_mar_time < 0 :
			ave_mar_time = default_marathon_time
		mrmt = most_recent_marathon_time(p)
		if mrmt <0 :
			mrmt = default_marathon_time
		p_features.append(mrmt)
		
		#p_features.append(average_marathon_imporvement(p))
		#print average_marathon_imporvement(p)
		#p_features.append(ave_mar_time)
	#	p_features.append(-num_marathons(p)*(300))
	#	p_features.append(variance_marathon_time(p,ave_mar_time))

		#Might want to change how we use the following features
		ave_pace = average_race_pace(p)
		#ave race pace
		p_features.append(ave_pace*42)
		#if age > 35 then assume that variance will occur with time gain and age < 35 time loss
		# if(p[0][6] >15):
		# 	age_coef = 1
		# else:
		# 	age_coef = -1
		
		#p_features.append(variance_marathon_time(p, ave_mar_time)*age_coef)
		p_features.append(variance_race_pace(p,ave_pace)*42 )

		#p_features.append(variance_race_pace(p,ave_pace))
		#age
		#p_features.append(p[0][6])
		#gender
		#p_features.append(p[0][7])
		if len(feature_sums) < 1 :
			feature_sums = [0]*len(p_features)
			feature_counts =[0]*len(p_features)
		for i in range(0,len(p_features)):
			if p_features[i] != -1:
				feature_sums[i] += p_features[i]
				feature_counts[i] += 1


		feature_set.append(p_features)
	#replace bad features with the averages
	for i  in range(0,len(feature_set)):
		for j in range(0,len(feature_set[i])):
			if feature_set[i][j] == -1:
				#This mess of casting is to ensure that binary variables stay binary 
				feature_set[i][j] = int(round(float(feature_sums[j])/feature_counts[j]))
#	print [float(feature_sums[x]/feature_counts[x]) for x in range(0,len(feature_sums))]
	return feature_set




######

'''
Long List of Methods used for creating features
'''
	 
#########

def average_marathon_imporvement(participant):
	sum = 0.0
	count = 0
	time_0 = 0.0
	time_1 = 0.0
	for race in participant:
		if race[4] == 42 and race[5]>0:
			time_1 = time_0
			time_0 = race[5]
			if time_1 > 0 :
				sum += time_1 - time_0
				count += 1
	if count == 0 :
		return -1
	return sum/count
def most_recent_marathon_time(participant):
	for race in participant:
		if race[4] == 42 and race[5]>0:
			
			return race[5]
	return -1

'''Find average distance, knowing distance is stored in 4th position of array'''
def average_distance(participant):
	distances = [race[4] for race in participant]
	#if np.mean(distances) > 42:
		#print distances
	return np.mean(distances)
'''Find variance, given the known mean and knowing dist is in 4th position'''
def variance_distance(participant, mean):
	sum = 0.0
	count = 0
	for race in participant:
		if(race[4]>-1):
			sum += (race[4] - mean) * (race[4] - mean)
			count += 1
	if count < 1:
		return -1
	return float(sum) / count

'''Find the most common distance ran by the given participant'''
def mode_distance(participant):
	distances = [race[4] for race in participant]
	return max(set(distances), key=distances.count)
'''Counts number of times that runner raced a marathon distance, 42 km'''
def num_marathons(participant):
	num = 0
	for race in participant:
		if race[4] == 42 and race[5]>0:
			num += 1
	return num

def average_marathon_time(participant):
	total_time = 0.0
	count = 0
	for race in participant:
		if race[4] == 42 and race[5]>0:
			total_time +=  race[5]
			count += 1
	if(count > 0):
		return float(total_time)/count
	return -1

def variance_marathon_time(participant,mean):
	sum = 0.0
	count = 0
	for race in participant:
		if race[4] == 42 and race[5]>0:
			sum += (race[5] - mean) * (race[5] - mean)
			count += 1
	if count >0:
		return float(sum)/count
	return -1

def average_race_pace(participant):
	sum = 0.0
	count = 0
	for race in participant:
		if race[4]>-1 and race[5]>0:
			sum += float(race[5])/race[4]
			count += 1
	if count > 0:
		return sum/count
	return -1

def variance_race_pace(participant,mean):
	sum = 0.0
	count = 0
	for race in participant:
		if race[4]>-1 and race[5]>0:
			sum += (float(race[5])/race[4] - mean) * (float(race[5])/race[4] - mean) 
			count += 1
	if count >0:
		return sum/count
	return -1

def number_of_montreal_marathons(participant):
	count = 0
	for race in participant:
		#in montreal and ran a marathon
		if race[3] == 1 and race[4] == 42:
			count += 1
	return count

def number_of_montreal_races(participant):
	count = 0
	for race in participant:
		if race[3] ==1:
			count += 1
	return count

'''Determine if the participant ran the marathon in montreal in the specific year '''	
def competed_in_montreal_marathon(participant,year):
	for race in participant:
		if race[0] == year and race[4] == 42:
			return 1
	return 0
'''Find the Avereage Days Between Each race they run'''	
def average_days_between_race(participant):
	num_races = len(participant)
	if num_races < 2: 
		return 300 # Default Value
	else:
		sum = 0.0
		count = 0
		for race_number in range(0,num_races -1):
			race_1 = participant[race_number] #More recent race
			race_2 = participant[race_number+1] #Next closest one 
 			sum+=days_between_dates(race_1[0],race_1[1],race_1[2], race_2[0],race_2[1],race_2[2])
 			count += 1
 		return sum/count

''' Finds the number of days between two dates given with seperate parameters'''
def days_between_dates(curr_year,curr_month,curr_day,prev_year,prev_month,prev_day):
	prev_date = date(prev_year,prev_month,prev_day)
	curr_date = date(curr_year,curr_month,curr_day)
	delta = curr_date - prev_date
	return int(delta.days)


'''Find days since the last race they ran approximates month length to 30.5 days'''
def days_since_last_race(participant,curr_year,curr_month,curr_day):
	race = participant[0]
	return  days_between_dates(curr_year,curr_month,curr_day,race[0],race[1],race[2])



if __name__ == "__main__" :
 	data = read_raw('./races.csv')
 	new_data = recast(data)
 	train_data , labels = split_data(2015,9,20,1,1,new_data, 0)
 	features = create_features_classification(train_data)



