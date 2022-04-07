import pandas as pd
# import autosklearn.classification
from tpot import TPOTClassifier
from sklearn.metrics import classification_report
import time

start_time = time.time()

train = pd.read_csv("CS-I/best_train.csv")
test = pd.read_csv("CS-I/best_test.csv")

print(train)

X_train, y_train = train[train.columns[1:(len(train.columns) - 1)]], train.iloc[:, -1]
X_test, y_test = test[test.columns[1:(len(test.columns) - 1)]], test.iloc[:, -1]

tpot = TPOTClassifier(generations=50, population_size=10, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

y_pred = tpot.predict(X_test)
print(classification_report(y_test, y_pred))

cost = (time.time() - start_time) / 60
print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)