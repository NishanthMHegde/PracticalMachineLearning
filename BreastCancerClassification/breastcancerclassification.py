from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 

#PROBLEM STATEMENT
"""
We have a dataset which consists of data related to breast tissues. The classifier model needs to make use of different features
of the breast tissue such as smoothness, radius, diameter, circumference, etc. to classify the tumor as benign (not dangerous) or 
malicious(serious).

"""
#Starting data preparation process
print("Starting data preparation process")
cancer = load_breast_cancer()
print("Cancer dataset has the keys")
print(cancer.keys())

print("The features available in the dataset are ")
print(cancer['feature_names'])


#let us create a pandas dataframe
df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ["target"]))

#The head of dataset is 
print(df.head())
#The tail of dataset is 
print(df.tail())
print(df.shape)
X =df.iloc[:, :30].values
y = df.iloc[:, 30].values

#Let us create our training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#Let us now normalize/ feature scale our data
"""
Formula for feature scaling:
X = (X- Xmin)/ (Xmax - Xmin)

The result is that we will have all independent variables whose value lies within the range [0,1]
"""
standard_scaler_X = StandardScaler()
X_train = standard_scaler_X.fit_transform(X_train)
X_test = standard_scaler_X.transform(X_test)

#Let us build our model
print("Starting Model Building and training phase")

classifier = SVC()
classifier = classifier.fit(X_train, y_train)

#Let us evaluate our model now
print("Starting Model Evaluation Phase")

y_pred = classifier.predict(X_test)

#use confusion matrix to check number of false positive or false negative
print("use confusion matrix to check number of false positive or false negative")
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix is ")
print(cm)

#Let us plot a seaborn plot to analyze the confusion matrix
print("Let us plot a seaborn plot to analyze the confusion matrix")
sns.heatmap(cm, annot = True) #set annot = True to check number of classifications in each category
plt.show()

#Let us check the accuracy of our predictions
print(classification_report(y_test, y_pred))

#We can see that the model has an accuracy of 96%. Let us try to maximise that
print("We can see that the model has an accuracy of 96%. Let us try to maximise that")


#Beginning model improvement
print("Beginning model improvement")

#Using GridSearchCV to find the right parameters and their values
print("Using GridSearchCV to find the right parameters and their values")

"""
How to use C values?

Smaller C : Whenever a misclassification occurs, the penalty levied on the misclassification is lesser. This ensures
that the model fitting is general and not too specific to the training data at hand.

Larger C value: Whenever a misclassification occurs, the penalty levied on the misclassification is high. This ensures
that the model tries to fit the training set too accurately and not too specific to the training data at hand. But the problem
with this larger C value is that it results in overfitting of model and since the model is not generic, it will misclassify the 
datapoints ccoming from the test dataset. 

"""
"""
How to use gamma values?

Smaller gamma : Tries to include maximu points on either side of the hyperplane instead of choosing only the points closer
to the hyperplane to classify the datapoints.

Larger gamma value: Includes only the points closer to the hyperplane and ignores the other points. The drawback of this apporach 
is that only the points which best describe the class it belongs to are used as benchmarks for future classification and this can lead
to missing out on variations of characterisits of points belonging to same class.

"""

parameters = {'C': [1, 10, 100, 0.1], 'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001]}

grid_search = GridSearchCV(param_grid = parameters,
							estimator = classifier,
							n_jobs= 1,
							cv=2,
							scoring='accuracy'
							)

grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
#the best parameters and their values are 
print("the best parameters and their values are")
print(best_params)

#Let us further check out different values for the best params
print("Let us further check out different values for the best params")
parameters = {'C': [100, 110, 120], 'kernel': ['rbf'], 'gamma': [0.001, 0.002, 0.003]}

grid_search = GridSearchCV(param_grid = parameters,
							estimator = classifier,
							n_jobs= 1,
							cv=2,
							scoring='accuracy'
							)
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
#the best parameters and their values are 
print("the best parameters and their values are")
print(best_params)

#Let us use the above values for C and gamma for our SVC classifier
print("Let us use the above values for C and gamma for our SVC classifier")

classifier = SVC(kernel='rbf', C=100, gamma=0.001)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#use confusion matrix to check number of false positive or false negative
print("use confusion matrix to check number of false positive or false negative")
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix is ")
print(cm)

#Let us plot a seaborn plot to analyze the confusion matrix
print("Let us plot a seaborn plot to analyze the confusion matrix")
sns.heatmap(cm, annot = True) #set annot = True to check number of classifications in each category
plt.show()

#Let us check the accuracy of our predictions
print(classification_report(y_test, y_pred))