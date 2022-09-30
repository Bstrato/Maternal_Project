import time

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
#print(y)


cs_delivered = data.loc[data['status'] == 0]
svd_delivered = data.loc[data['status'] == 1]
# print(f"RTSS1 TAKEN = {len(rtss_taken)}")
# print(f"RTSS1 NOT TAKEN = {len(rtss_not_taken)}")

X = data.drop("status", axis=1)
y = data['status']
#print(X)



smt = SMOTE(random_state=2)

#spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train_res, y_train_res = smt.fit_resample(X, y.ravel())

#X_train.shape, X_test.shape, y_train.shape, y_test.shape
#w = y_train.value_counts()
#print(w)

feature_names = [f"feature {i}" for i in range(X.shape[1])]
classifier = RandomForestClassifier()
b = classifier.fit(X_train_res, y_train_res.ravel())
pred = classifier.predict(X_test)
#print(pred)


from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(classifier, X_train_res, y_train_res)
elasped_time = time.time() - start_time
print(f"Elapsed time to compute the importance: {elasped_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature Importances")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()




# Checking the actual values against the predicted values
chk = pd.DataFrame(np.c_[y_test, pred], columns=["Actual", "Predicated"])
#print(chk)

cm = confusion_matrix(y_test, pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()

# Checking the accuracy of model
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))


