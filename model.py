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

# Load the dataset from the CSV file

data = pd.read_csv('Datasets/GymTrackingDataset.csv')

# Select features and target
features = data[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Fat_Percentage']]
target = data['Workout_Type']

# encode gender
gender_encoder = LabelEncoder()
features['Gender'] = gender_encoder.fit_transform(features['Gender'])

# Encode target labels (Workout_Type)
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Initialize the models to test
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Evaluate each model and store results
accuracy_scores = {}
classification_reports = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and store the results
    accuracy_scores[model_name] = accuracy_score(y_test, y_pred)
    classification_reports[model_name] = classification_report(y_test, y_pred, target_names=label_encoder.classes_,
                                                               output_dict=True)

# Plotting the accuracy of each model
plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(),
        color=['skyblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()

# Print classification reports for all models
for model_name, report in classification_reports.items():
    print(f"\nClassification Report for {model_name}:")
    print(pd.DataFrame(report).transpose())


# Function to predict workout type for new input
def predict_workout(age, gender, weight, height, fat_percentage, model):
    input_features = [[age, gender, weight, height, fat_percentage]]
    prediction = model.predict(input_features)
    return label_encoder.inverse_transform(prediction)[0]


# Example prediction using Decision Tree
model = models['Decision Tree']
example_prediction = predict_workout(age=59, gender=0, weight=150, height=1.7, fat_percentage=70, model=model)
print("\nRecommended Workout Type (using ", model, "):", example_prediction)
