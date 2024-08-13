import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv('./data/pima-indians-diabetes.data.csv',
                   names=header)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X,Y, test_size= 0.3)

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

acc= accuracy_score(Y_test, y_pred)
print(acc)
