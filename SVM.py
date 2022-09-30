import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, accuracy_score, \
    ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE




data = pd.read_excel("keziah_final.xlsx")
s = data.head()
y = data['status'].value_counts()
# print(y)


cs_delivered = data.loc[data['status'] == 1]
svd_delivered = data.loc[data['status'] == 2]
# print(f"RTSS1 TAKEN = {len(rtss_taken)}")
# print(f"RTSS1 NOT TAKEN = {len(rtss_not_taken)}")

X = data.drop("status", axis=1)
y = data['status']
#print(X)

smt = SMOTE(random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


X_train_res, y_train_res = smt.fit_resample(X, y.ravel())
#X_train.shape, X_test.shape, y_train.shape, y_test.shape
#w = y_train.value_counts()
#q = X_train.value_counts()
print(X_train_res, y_train_res)


classifier = linear_model.SGDClassifier(tol=1e-3, verbose=0)
#SVC(kernel='linear')
b = classifier.fit(X_train_res, y_train_res.ravel())
pred = classifier.predict(X_test)
# print(accuracy_score(y_test,y_pred))

chk = pd.DataFrame(np.c_[y_test, pred], columns=["Actual", "Predicated"])
# print(chk)

cm = confusion_matrix(y_test, pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()

#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


