import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Datasets/GymTrackingDataset.csv')

# Select X Y
X = data[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Fat_Percentage']]
Y = data['Workout_Type']

# encode gender so that it works with integers
gender_encoder = LabelEncoder()
X.loc[:, 'Gender'] = gender_encoder.fit_transform(X['Gender'])


# Encode workout type
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(Y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target_encoded, test_size=0.2, random_state=42)

# Select models to test
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

model = models['Decision Tree']

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Function to predict workout type for new input
def predict_workout(age, gender, weight, height, fat_percentage):
    input_features = [[age, gender, weight, height, fat_percentage]]
    prediction = model.predict(input_features)
    return label_encoder.inverse_transform(prediction)[0]

