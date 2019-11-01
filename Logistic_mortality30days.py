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

# Compute ROC curve and ROC area 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


#Plot of a ROC curve
data_X, class_label = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

probs = classifier.predict_proba(X_test)
probs =probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)





