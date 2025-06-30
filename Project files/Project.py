
# Sample features: you can expand as needed
features = [
    'hour', 'day_of_week', 'temperature', 'precipitation', 
    'is_holiday', 'road_type', 'lane_count'
]
target = 'traffic_volume'



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['road_type'], drop_first=True)

# Feature selection
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae:.2f} vehicles")
---

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(predictions[:100], label='Predicted')
plt.legend()
plt.title('Traffic Volume Estimation')
plt.xlabel('Sample Index')
plt.ylabel('Vehicle Count')
plt.show()


---
