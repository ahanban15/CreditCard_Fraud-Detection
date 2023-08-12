import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
import pickle


data = pd.read_csv("creditcard.csv")
print(data.info())

target = data["Class"]
data = data.drop('Class', axis=1)

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# rfc = RandomForestClassifier(n_estimators=250)
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
# print(rfc.fit(x_train, y_train))

predict = rfc.predict(x_test)
pickle.dump(rfc, open("creditcard1.pkl", "wb"))

# print(predict)
# print(rfc.score(x_train, y_train))
# print(accuracy_score(y_test, predict))
