import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

data = fetch_california_housing()

# splitting the train, test data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,test_size=0.2, random_state=42)

# Scale the train, test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model fitting
model = RandomForestRegressor(n_estimators=100, random_state=42)
# print(model)
model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)

# # model Evaluation
# accuracy = accuracy_score(y_test, y_pred)

# print('Accuracy score of the model is: {accuracy}')

joblib.dump(model, 'model.pkl')
print('Model trained and saved to model.pkl')
