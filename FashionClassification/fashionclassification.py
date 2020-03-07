from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Sequential 
from keras.optimizers import Adam 
from sklearn.metrics import confusion_matrix, classification_report 

"""
The dataset consists of samples of images of fashion items which are of the dimensions (28 x 28). The pixel
values of these 28 * 28 rows and columns are provided in the dataset. 
Each pixel can have values between 0 to 255. 0 means fully black and 255 means fully white.
All the 28 * 28 pixel values are put into one row which signifies one sample.
The label values or the items that will be used for classification are as follows:
Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

"""
train = pd.read_csv('fashion-mnist_train.csv', sep=',')
test = pd.read_csv('fashion-mnist_test.csv', sep=',')

#Convert the pandas dataframe into a numpy array of float datatype.
training = np.array(train, dtype='float32')
testing = np.array(test, dtype='float32')

# A sample image can be viewed by reshaping it to 28 * 28 on matplotlib.
# plt.imshow(training[2,1:].reshape(28,28))
# plt.show()

#We segrate out training/testing data into X and y.
#We also apply Feature scaling to have all values between 0 and 1. Best way is to divide the pixel value by 255, which gives value in range[0,1]
X_train = training[:,1:]/255
y_train = training[:,0]


X_test = testing[:,1:]/255
y_test = testing[:,0]

#We now need to create a training and validation dataset within our training dataset.
#Validation data will be used for validating the CNN model as we train it to calculate the accuracy.

X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train, test_size=0.2, random_state=0)

"""
We reshape the train, test and validation dataset to make it have 3 dimensions, namely:
X = number of samples,
Y = row pixel,
z = column pixel
"""
X_train = X_train.reshape(X_train.shape[0],*(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0],*(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0],*(28, 28, 1))

print("The shape of the train, validate and test datasets are:")
print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)

#We construct the sequential model CNN
cnn_model = Sequential()

#We apply convolution using 32 feature detectors each having 3*3 grid size and taking one stride at a time.
#We also use the relu activation function
cnn_model.add(Conv2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu'))
#We apply max pooling with a pool_size of a 2*2 matrix to get our max pooled feature maps
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

#We flatten the max pooled matrix to a 1D flattened out matrix
cnn_model.add(Flatten())

#We add our first deep neural network layer with a relu activation function which takes in the 32 flattened arrays from each feature deector
cnn_model.add(Dense(output_dim=32, init='uniform', activation='relu'))
# We add another deep neural layer with sigmoid activation function for classification as output layer
cnn_model.add(Dense(output_dim=10, init='uniform', activation='sigmoid'))

#We compile the model and use sparse_categorical_crossentropy instead of binary crossentropy as we are dealing with multiple labels insted of 2 labels.
cnn_model.compile(loss = "sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

#We fit our model to our training dataset and validate it against our validation dataset to detect accuaracy.

#Batch size indicates how many samples will be used in each epoch. After every epoch, the weights are adjusted using back propogation.
cnn_model.fit(X_train, y_train, batch_size=512, nb_epoch=30, verbose=1, validation_data=(X_validate, y_validate))

#We evaluate our model against the test dataset to get to know accuracy on the TEST dataset.
evaluation_metrics = cnn_model.evaluate(X_test, y_test)
print("The evaluation details from test data are")
print(evaluation_metrics)

print("Accuracy on test data is %s" %(float(evaluation_metrics[1]*100)))
print(evaluation_metrics)
predicted_classes = cnn_model.predict_classes(X_test)

cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True)
plt.show()

class_labels = list()
for i in range(1, 11):
	class_labels.append("Class {%s}" %(i))

print("The classification report is ")

print(classification_report(y_test, predicted_classes, target_names = class_labels))

#Model improvement can be done in the following ways:
"""
1. Adding more feature detectors. Instead of 32, we can use 64 feature detectors.
2. Varying the number of hidden neural network layers.
3. Adding a Dropout. A dropout reduces the interconnectedness between neurons to some extent. This prevents over-fitting and can give good results.

"""

