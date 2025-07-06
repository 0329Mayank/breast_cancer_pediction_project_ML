
from data_processing import load_data
from model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

df = load_data()
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
