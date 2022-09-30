import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, accuracy_score, \
    ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

data = pd.read_excel("keziah_final.xlsx")
s = data.head()
y = data['status'].value_counts()
print(y)

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

# X_train.shape, X_test.shape, y_train.shape, y_test.shape
# w = y_train.value_counts()
print(X_train_res, y_train_res)

classifier = RandomForestClassifier()
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

# Checking the accuracy of model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# roc curve for classes
fpr = {}
tpr = {}
thresh = {}
roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    # plotting
    plt.plot(fpr[i], tpr[i], linestyle='--',
             label='%s vs Rest (AUC=%0.2f)' % (classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'b--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='lower right')
plt.show()
