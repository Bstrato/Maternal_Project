import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, accuracy_score, \
    ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE


data = pd.read_excel("keziah_final.xlsx")
s = data.head()
y = data['status'].value_counts()
# print(y)


cs_delivered = data.loc[data['status'] == 0]
svd_delivered = data.loc[data['status'] == 1]
# print(f"RTSS1 TAKEN = {len(rtss_taken)}")
# print(f"RTSS1 NOT TAKEN = {len(rtss_not_taken)}")

X = data.drop("status", axis=1)
y = data['status']
# print(X)


smt = SMOTE(random_state=42)


# spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train_res, y_train_res = smt.fit_resample(X, y.ravel())
#X_train.shape, X_test.shape, y_train.shape, y_test.shape
#w = y_train.value_counts()
print(X_train_res, y_train_res)


classifier = GaussianNB()
b = classifier.fit(X_train_res, y_train_res.ravel())
pred = classifier.predict(X_test)
# print(pred)


# Checking the actual values against the predicted values
chk = pd.DataFrame(np.c_[y_test, pred], columns=["Actual", "Predicated"])
# print(chk)


cm = confusion_matrix(y_test, pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()

#Computing AUC
auc = roc_auc_score(y_test, pred)

# Checking the accuracy of model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

print('AUC: %.2f' % auc)


g = sns.countplot(data['status'])
g.set_xticklabels(['Unsuccessful', 'Delivered'])
plt.show()
