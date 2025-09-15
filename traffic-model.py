import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

# Dummy data: [time, day, weather, current_traffic] -> congestion (0=low, 1=medium, 2=high)
X = np.array([
    [8, 1, 0, 50], [12, 1, 1, 70], [17, 1, 0, 90],
    [8, 5, 0, 40], [12, 5, 2, 60], [17, 5, 1, 80]
] * 10)
y = np.array([1, 2, 2, 0, 1, 2] * 10)

# Train the AI model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict traffic
def predict_traffic(time, day, weather, current):
    input_data = np.array([[time, day, weather, current]])
    return model.predict(input_data)[0]

# Allow command-line input
if _name_ == '_main_':
    if len(sys.argv) == 5:
        time, day, weather, current = map(float, sys.argv[1:])
        # print(predict_traffic(time, day, weather,current))
