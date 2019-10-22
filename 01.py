import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#csv is short for comma separated values
#reading the data set file into the main src
data = pd.read_csv("student-mat.csv", sep=";")

#here, we only use a couple of these 33 attributes
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)

#to find the line with the highest accuracy, using a for loop to loop thru a particular ammount of times  to find the best line
best = 0
for _ in range(30):
	#now we'll split the above two arrays in 2, one for trainig and the other for testing the model
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)

	linear = linear_model.LinearRegression()

	linear.fit(x_train, y_train)
	acc = linear.score(x_test, y_test)
	print(acc)

	if acc > best: #we will write to the file if a better line accuracy is found
		#Saving the trained model to use for future data sets
		best = acc 
		with open("studentmodel.pickle", "wb") as f:
			pickle.dump(linear, f)
	#saves a pickle file which has the model in it


pickle_in = open("studentmodel.pickle", "rb")	
linear = pickle.load(pickle_in)


#Now to find the constants m and c for the best fit line y = mx + c
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)


#now lets use this to predict for a student's final marks
predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print(predictions[x], x_test[x], y_test[x])

#plotting a graph using matplotlib
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()