# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('MortalityClass.csv')
#X = dataset.iloc[:, [10,11,12, 13]].values

X = dataset.iloc[:, [2,3,5,7,8,9]].values
y = dataset.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ############        Logistic Regression                 ###########


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier, X= X_train, y = y_train, cv=5)
accuracy.mean()
accuracy.std()


                    # K-Nearest Neighbors (K-NN)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
KNNcm = confusion_matrix(y_test, y_pred_KNN)
KNNacc = accuracy_score(y_test, y_pred_KNN)


        # Support Vector Machine (SVM)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_SVM = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
SVMcm = confusion_matrix(y_test, y_pred_SVM)
SVMacc = accuracy_score(y_test, y_pred_SVM)


# Kernel SVM 

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KSVM = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
KERSVMcm = confusion_matrix(y_test, y_pred_KSVM)
KERSVMacc = accuracy_score(y_test, y_pred_KSVM)


         # Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 25, criterion ='entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score
RFcm = confusion_matrix(y_test, y_pred_RF)
RFacc = accuracy_score(y_test, y_pred_RF)