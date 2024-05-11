'''Random Forest Training and Analysis'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

FILEPATH=""

mnist_train = pd.read_csv(FILEPATH + "MNIST_train (1).csv").drop("Unnamed: 0", axis = 1)
mnist_test = pd.read_csv(FILEPATH + "MNIST_test (1).csv").drop("Unnamed: 0", axis = 1)

#Separate predictors and labels for traininng & testing
X_train = mnist_train.drop('response', axis = 1)
y_train = mnist_train['response']
X_test = mnist_test.drop('response', axis = 1)
y_test = mnist_test['response']

#Fit rfc with 5 estimators
model1 = rfc(n_estimators = 5, random_state = 419)
model1.fit(X_train, y_train)

#Fit rfc with 5 estimators and max depth 4
model2 = rfc(n_estimators = 5, max_depth = 4, random_state = 419)
model2.fit(X_train, y_train)

#Fit rfc with 5 estimators with pruning regularization
model3 = rfc(n_estimators = 5, ccp_alpha = 0.01, random_state = 419)
model3.fit(X_train, y_train)

#Fit rfc with 50 estimators and max depth 10
model4 = rfc(n_estimators = 50, max_depth = 10, random_state = 419)
model4.fit(X_train, y_train)

#Fit rfc with 50 estimators with pruning regularization
model5 = rfc(n_estimators = 50, ccp_alpha = 0.001, random_state = 419)
model5.fit(X_train, y_train)

#fit Gradient Boosting Classifier
model6 = gbc(random_state = 419)
model6.fit(X_train, y_train)

#model 1 confusion matrix
predictions1 = model1.predict(X_test)
disp1 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions1,
                                               cmap=plt.cm.Blues)
disp1.ax_.set_title('Model 1 Confusion Matrix');
plt.show()

#model 2 confusion matrix
predictions2 = model2.predict(X_test)
disp2 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions2,
                                               cmap=plt.cm.Blues)
disp2.ax_.set_title('Model 2 Confusion Matrix');
plt.show()

#model 3 confusion matrix
predictions3 = model3.predict(X_test)
disp3 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions3,
                                               cmap=plt.cm.Blues)
disp3.ax_.set_title('Model 3 Confusion Matrix');
plt.show()

#model 4 confusion matrix
predictions4 = model4.predict(X_test)
disp4 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions4,
                                               cmap=plt.cm.Blues)
disp4.ax_.set_title('Model 4 Confusion Matrix');
plt.show()

#model 5 confusion matrix
predictions5 = model5.predict(X_test)
disp5 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions5,
                                               cmap=plt.cm.Blues)
disp5.ax_.set_title('Model 5 Confusion Matrix');
plt.show()

#model 6 confusion matrix
predictions6 = model6.predict(X_test)
disp6 = ConfusionMatrixDisplay.from_predictions(y_test,
                                               predictions6,
                                               cmap=plt.cm.Blues)
disp6.ax_.set_title('Model 6 Confusion Matrix');
plt.show()

#Gradient boosing accuracy by number
zero_acc = confusion_matrix(y_test, predictions6)[0][0] / sum(confusion_matrix(y_test, predictions6)[0])
five_acc = confusion_matrix(y_test, predictions6)[1][1] / sum(confusion_matrix(y_test, predictions6)[1])
six_acc = confusion_matrix(y_test, predictions6)[2][2] / sum(confusion_matrix(y_test, predictions6)[2])
eight_acc = confusion_matrix(y_test, predictions6)[3][3] / sum(confusion_matrix(y_test, predictions6)[3])

viz_lst = [zero_acc, five_acc, six_acc, eight_acc]
num_lst = ['0', '5', '6', '8']

plt.bar(num_lst, viz_lst)
plt.ylim(0.9, 1.0)
plt.ylabel('Accuracy')
plt.xlabel('Number')
plt.title('Accuracy of the Gradient Boosting Classifier Model')
plt.show()