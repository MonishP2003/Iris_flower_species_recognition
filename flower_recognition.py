import numpy as np
import pickle

# Load the model
model = pickle.load(open('iris_model', 'rb'))
# Get the flower characteristics to test the model
test = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction = model.predict(test)
# Predict the flower species using the model
print("Prediction of Species: {}".format(prediction))
