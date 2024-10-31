from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



#open document
file_path = "spam.csv"
#standard enconding delivered an error, so using differend encoding
df = pd.read_csv(file_path, encoding='ISO-8859-1')

#Remove empty colums
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#Make x and y
texts = df['v2']
y = df['v1']

#Transform Text to vector
vector = CountVectorizer(binary=True)
X = vector.fit_transform(texts)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

#Benoulli's naive bayes
BNB = BernoulliNB()
BNB.fit(X_train, y_train)

#make prediction
y_pred = BNB.predict(X_test)

#Test acc
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')
