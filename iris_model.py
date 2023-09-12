import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pickle

# Import the iris dataset
iris = pd.read_csv(r'Dataset file path')
# Get the characteristics of each species
setosa = iris[iris.Species == "Iris-setosa"]
versicolor = iris[iris.Species == 'Iris-versicolor']
virginica = iris[iris.Species == 'Iris-virginica']
fig, ax = plt.subplots()
fig.set_size_inches(13, 7)

# Plot the petal characteristics on a scatterplot
ax.scatter(setosa['PetalLengthCm'], setosa['PetalWidthCm'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], label="Versicolor", facecolor="green")
ax.scatter(virginica['PetalLengthCm'], virginica['PetalWidthCm'], label="Virginica", facecolor="red")

ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

# Trainthe model using logistic regression
X = iris.drop(['Species'], axis=1)
X = X[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
training_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)
# Analyse the metrics for model performance
print("Precision, Recall, Confusion matrix, in training\n")
print(metrics.classification_report(y_train, training_prediction, digits=3))
CM = metrics.confusion_matrix(y_train, training_prediction)
print(CM)
print("True negatives: ", CM[0][0])
print("False negatives: ", CM[1][0])
print("True positives: ", CM[1][1])
print("False positives: ", CM[0][1])

# Save the model
f_name = 'iris_model'
pickle.dump(model, open(f_name, 'wb'))
