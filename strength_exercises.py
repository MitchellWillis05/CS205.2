import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load the dataset
dataset_path = 'Datasets/StrengthExerciseDataset.csv'
df = pd.read_csv(dataset_path)

# Encode categorical variables to numerical values
label_encoders = {}
for column in ["Type", "BodyPart", "Equipment"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Features definition
# Drops rows if the "Rating" column is N/A
df = df.dropna(subset=["Rating"])
X = df[["Type", "BodyPart", "Equipment"]]
y = df["Rating"]


# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Function to recommend exercises based on preferences
def recommend_exercises(preferred_type, preferred_body_part, preferred_equipment):
    # Encode preferences
    type_encoded = label_encoders["Type"].transform([preferred_type])[0]
    body_part_encoded = label_encoders["BodyPart"].transform([preferred_body_part])[0]
    equipment_encoded = label_encoders["Equipment"].transform([preferred_equipment])[0]

    input_data = pd.DataFrame(
        np.array([[type_encoded, body_part_encoded, equipment_encoded]]),
        columns=["Type", "BodyPart", "Equipment"]
    )
    predicted_rating = model.predict(input_data)[0]

    # Find the best matching exercise
    filtered_df = df[
        (df["Type"] == type_encoded) &
        (df["BodyPart"] == body_part_encoded) &
        (df["Equipment"] == equipment_encoded)
        ]
    best_exercise = filtered_df.loc[filtered_df["Rating"].idxmax()]

    return {
        "Title": best_exercise["Title"],
        "Description": best_exercise["Desc"],
        "Rating": best_exercise["Rating"]
    }


# Example usage
# preferred type, preferred body part, preferred equipment
p_type = "Strength"
p_body_part = "Chest"
p_equipment = "Cable"

recommendation = recommend_exercises(p_type, p_body_part, p_equipment)
print("\nRecommended Exercise:")
print(recommendation)
