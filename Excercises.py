import pandas as pd

# Load the dataset
dataset_path = 'Datasets/ExerciseDataset.csv'
df = pd.read_csv(dataset_path)

# Display initial rows of the dataset
print("Dataset Preview:")
print(df.head())


# Function to sort exercises by a specified column
def sort_exercises(df, column_name, ascending=True):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    return df.sort_values(by=column_name, ascending=ascending)


# Example sorting operations
sorted_by_rating = sort_exercises(df, "Rating", ascending=False)  # Highest-rated exercises first
sorted_by_level = sort_exercises(df, "Level")  # Beginner to advanced levels


# Function to filter exercises based on criteria
def filter_exercises(df, body_part=None, equipment=None, level=None):
    filtered_df = df.copy()
    if body_part:
        filtered_df = filtered_df[filtered_df["BodyPart"].str.contains(body_part, case=False, na=False)]
    if equipment:
        filtered_df = filtered_df[filtered_df["Equipment"].str.contains(equipment, case=False, na=False)]
    if level:
        filtered_df = filtered_df[filtered_df["Level"].str.contains(level, case=False, na=False)]

    return filtered_df


# Example filtering operations
upper_body_exercises = filter_exercises(df, body_part="Upper Body")
dumbbell_exercises = filter_exercises(df, equipment="Dumbbell")
beginner_exercises = filter_exercises(df, level="Beginner")


# Function to group exercises by a column
def group_exercises(df, group_by_column):
    if group_by_column not in df.columns:
        raise ValueError(f"Column '{group_by_column}' not found in the dataset.")

    grouped = df.groupby(group_by_column).size().reset_index(name='Count')
    return grouped


# Example grouping operation
grouped_by_body_part = group_exercises(df, "BodyPart")
print("\nExercises Sorted by Rating:")
print(sorted_by_rating)
