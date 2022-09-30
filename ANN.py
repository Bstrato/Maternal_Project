import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

data = pd.read_excel("keziah_final.xlsx")
s = data.head()
y = data['status'].value_counts()
# print(y)


cs_delivered = data.loc[data['status'] == 0]
svd_delivered = data.loc[data['status'] == 1]

X = data.drop("status", axis=1)
y = data['status']

smt = SMOTE(random_state=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_res, y_train_res = smt.fit_resample(X, y.ravel())

# b = classifier.fit(X_train_res, y_train_res.ravel())

sc = StandardScaler()
X_train = sc.fit_transform(X_train_res, y_train_res.ravel())
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=32, activation="relu"))
ann.add(tf.keras.layers.Dense(units=16, activation="relu"))
ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="softmax"))
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
ann.fit(X_train_res, y_train_res, batch_size=10, epochs=50)

y_pred = sc.predict(X_test)
y_pred = (y_pred > 0.5)

# print(y_pred)


cm = tf.maths.confusion_matrix(y_test, pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
print(cm)
