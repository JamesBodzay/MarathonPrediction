# MarathonPrediction
Predict Performance of previous racers in Montreal Marathon in the 2016 Montreal Marathon
To run predictions : ./run.py

./run.py will print out validation error, train error from the test  as well as print final test predictions to a file. </br>
parse_data contains methods used to read data from a csv file and convert into features usable by linear regressor </br>
validator contains methods to split data into validation sets inorder to predict accuracy of our learned models.
