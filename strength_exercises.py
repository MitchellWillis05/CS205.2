import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_path = 'Datasets/StrengthExerciseDataset.csv'
df = pd.read_csv(dataset_path)

# Encode categorical variables to numerical values
label_encoders = {}
for column in ["Type", "BodyPart", "Equipment"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Drop rows where "Rating" is missing
df = df.dropna(subset=["Rating"])


# Function to recommend exercises based on preferences
def recommend_exercises(preferred_type, preferred_body_part, preferred_equipment):
    # Encode preferences
    type_encoded = label_encoders["Type"].transform([preferred_type])[0]
    body_part_encoded = label_encoders["BodyPart"].transform([preferred_body_part])[0]
    equipment_encoded = label_encoders["Equipment"].transform([preferred_equipment])[0]

    # Filter exercises based on preferences
    filtered_df = df[
        (df["Type"] == type_encoded) &
        (df["BodyPart"] == body_part_encoded) &
        (df["Equipment"] == equipment_encoded)
    ]

    # If exact matches are found, recommend the highest-rated exercise
    if not filtered_df.empty:
        best_exercise = filtered_df.loc[filtered_df["Rating"].idxmax()]
        return {
            "Title": best_exercise["Title"],
            "Description": best_exercise["Desc"],
            "Rating": best_exercise["Rating"]
        }

    # If no exact match is found
    return {
        "Title": "No exact match found.",
        "Message": "Try modifying your preferences or explore other options."
    }

