import pandas as pd
import quandl, math, datetime

import numpy as np
# from sklearn import preprocessing, cross_validation, svm
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style



style.use('ggplot')
dataFrame = quandl.get('WIKI/GOOGL')
# dataFrame.shape: (3114, 12)
# dataFrame.head(): This command prints out the columns and rows neatly on terminal so you can see what you r working on more clearly

dataFrame = dataFrame[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

############	DEFINE YOUR FEATURES	############
# dataFrame.shape:(3114, 5)
dataFrame['HL_PCT'] = (dataFrame['Adj. High'] - dataFrame['Adj. Low']) / dataFrame['Adj. Low'] * 100 # 1st feature: (High - Low) / Low
dataFrame['PCT_change'] = (dataFrame['Adj. Close'] - dataFrame['Adj. Open']) / dataFrame['Adj. Open'] * 100 # 2nd feature: 
dataFrame = dataFrame[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] # Take only the features that we actually care!

forecast_column = 'Adj. Close'			# Adj. Close is gonna be used for future prediction
dataFrame.fillna(-9999, inplace = True) # If there are NaN anywhere in the data, 
										# replace it with -9999 to be treated as an outlier

# len(dataFrame): 3114
forecast_out = int(math.ceil(0.01*len(dataFrame))) # Predict using %1 of your past values. i.e if len(dataFrame) returns 100, you will predict into the future using past 1 days prices
# print forecast_out
dataFrame['label'] = dataFrame[forecast_column].shift(-forecast_out) # The 'label' column for each row will be the 'Adj. Close' price %1 days into the future
dataFrame.dropna(inplace = True)

X = np.array(dataFrame.drop(['label'],1)) 	# Features are everything except the whole 'label' column
y = np.array(dataFrame['label']) 			# y = The whole 'label' column 

X = preprocessing.scale(X)  # Scale function scales your features between [-1,1],helping the processing speed
							# It is all scaled together so it is normalized with all the other 
							# data points so in order to properly scale it, you would have to
							# include your training data. So keep that in mind if you do this 
							# you need to scale the new values, but not just scale them, but scale
							# them alongside your other values. While this can help with training 
							# and testing, it can add processing time. If you r doin high freq. trading
							# you would almost certainly skipped this step.



y = np.array(dataFrame['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 )# It is going to take
																							# our features and labels
																							# shuffle them up while keeping 
																							# Xs' and ys' still connected
																							# and outputs to X_train <-> y_train
																							# X_test <-> y_test
#	classifier = svm.SVR(): did not perform well
classifier = LinearRegression() 			# Define your classifier
classifier.fit(X_train,y_train) 			# We use train set to fit our data to our classifier.
 											# We can use this classifier to predict into the future
accuracy = classifier.score(X_test,y_test)	# But first, we should probably test it right? and see what our accurcy is
											# Accuracy is the squared error,cuz we used Linear Regression.
print('Accuracy: ', accuracy)
 










