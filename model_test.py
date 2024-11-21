import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset
data = pd.read_csv('Datasets/GymTrackingDataset.csv')

# Select X and Y
X = data[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Fat_Percentage']]
Y = data['Workout_Type']

# Encode gender so that it works with integers
gender_encoder = LabelEncoder()
X['Gender'] = gender_encoder.fit_transform(X['Gender'])


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

# Train each model and evaluate performance
results = {}
confusion_matrices = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

# Plot the accuracy of each model
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = list(results.values())
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Accuracy Scores')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('model_performance', 'model_accuracies.png'))
plt.show()

# Visualize confusion matrix for each model
for model_name, cm in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join('model_performance', f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'))
    plt.show()
