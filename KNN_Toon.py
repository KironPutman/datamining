import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#import data
url = "https://raw.githubusercontent.com/businessdatasolutions/courses/main/data%20mining/gitbook/datasets/breastcancer.csv"
df = pd.read_csv(url)

#transform M and D into labels
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])


# make X and y (x = numeric data, y = the outcome)
X = df.drop(columns = ['diagnosis'])
y = df['diagnosis']

#split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

#Normilize the data via min max (Makes al the data between 0 and 1)
scaler = MinMaxScaler()
X_trained_norm = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#run knn on data
knn = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree')
#fit data (create data line)
knn.fit(X_trained_norm, y_train)

#measure accuracy
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

#print results
print(acc)
print(f"Accuracy: {acc * 100:.1f}%")