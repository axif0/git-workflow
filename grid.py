from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logit = LogisticRegression(max_iter = 10000)

print(logit.fit(X_train, y_train))

print(logit.score(X_test, y_test))
