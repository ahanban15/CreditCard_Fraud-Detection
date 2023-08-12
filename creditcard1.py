import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef

data = pd.read_csv("creditcard.csv")
print(data.info())
print(data.shape)
print(data.describe())


print('No. of Credit Card FRAUD Transactions: {}'.format(len(data[data['Class'] == 1])))
print('No. of Credit Card VALID Transactions: {}'.format(len(data[data['Class'] == 0])))

# plt.plot(data['Time'])
# plt.xlabel('Transactions')
# plt.ylabel('Time')
# plt.title('Time Transaction Line PLot')
# plt.show()

# plt.plot(data['Amount'])
# plt.xlabel('Transactions')
# plt.ylabel('Amount')
# plt.title('Amount Transaction Line PLot')
# plt.show()

# plt.plot(data['Class'])
# plt.xlabel('Transactions')
# plt.ylabel('Class [Fraud(1)/Valid(0)]')
# plt.title('Class Transaction Line PLot')
# plt.show()

# corrmat = data.corr()
# fig = plt.figure(figsize=(15, 9))
# sns.heatmap(corrmat, vmax=.5, square=True)
# plt.title('Correlation Matrix Heatmap')
# plt.show()

# absolute values of columns to get scatter plot and histogram
# graph_data = data.copy()
# graph_data['V1'] = graph_data["V1"].apply(lambda x: int(-1) * x if x < 0 else int(x)).values

# plt.scatter(graph_data['Time'], graph_data['V1'], color='green')
# plt.scatter(graph_data['Time'], graph_data['V10'], color='red')
# plt.scatter(graph_data['Time'], graph_data['V15'], color='blue')
# plt.scatter(graph_data['Time'], graph_data['V20'], color='yellow')
# plt.xlabel('Transaction')
# plt.ylabel('PCA Features')
# plt.title('PCA features Transaction Scatter Plot')
# plt.show()

# plt.hist(graph_data['V1'], rwidth=0.8)
# plt.xlabel('Transaction')
# plt.ylabel('PCA Feature V1')
# plt.title('PCA feature V1 Transaction Histogram')
# plt.show()

target = data["Class"]
data = data.drop('Class', axis=1)

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# rfc = RandomForestClassifier(n_estimators=250)
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
# print(rfc.fit(x_train, y_train))

predict = rfc.predict(x_test)
# predict = rfc.predict([[472,-3.0435406239976,-3.15730712090228,1.08846277997285,2.2886436183814,1.35980512966107,-1.06482252298131,0.325574266158614,-0.0677936531906277,-0.270952836226548,-0.838586564582682,-0.414575448285725,-0.503140859566824,0.676501544635863,-1.69202893305906,2.00063483909015,0.666779695901966,0.599717413841732,1.72532100745514,0.283344830149495,2.10233879259444,0.661695924845707,0.435477208966341,1.37596574254306,-0.293803152734021,0.279798031841214,-0.145361714815161,-0.252773122530705,0.0357642251788156,529]])
print(predict)

# printing the confusion matrix
LABELS = ['Valid', 'Fraud']
conf_matrix = confusion_matrix(y_test, predict)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

print(rfc.score(x_train, y_train))
# print(accuracy_score(y_test, predict))

n_errors = (predict != y_test).sum()
print("The model used is Random Forest classifier")

acc = accuracy_score(y_test, predict)
print("The accuracy is {}".format(acc))

prec = precision_score(y_test, predict)
print("The precision is {}".format(prec))

rec = recall_score(y_test, predict)
print("The recall is {}".format(rec))

f1 = f1_score(y_test, predict)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(y_test, predict)
print("The Matthews correlation coefficient is {}".format(MCC))
