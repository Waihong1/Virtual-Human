import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import os

h2o.init()

# Get the current script's directory (this will work even if cloned)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the CSV file
csv_path = os.path.join(script_dir, 'user_avatar_choices.csv')

# Read the CSV file using the dynamic path
df = pd.read_csv(csv_path)

df = pd.get_dummies(df, columns=["user_gender", "user_skin_tone", "question_id"])

# Convert chosen avatar to binary (0 for A, 1 for B)
df["chosen_avatar"] = (df["chosen_avatar"] == df["avatar_B_id"]).astype(int)

# Convert to H2O Frame
data = h2o.H2OFrame(df)

# Define input features and target
x = [col for col in data.columns if col not in ["chosen_avatar", "avatar_A_id", "avatar_B_id"]]
y = "chosen_avatar"

# Convert target to categorical (for classification)
data[y] = data[y].asfactor()

# Train model using H2O AutoML
aml = H2OAutoML(max_models=20, seed=42)
aml.train(x=x, y=y, training_frame=data)

# View leaderboard
print(aml.leaderboard)

leaderboard = aml.leaderboard

# Get the best model
best_model_id = leaderboard[0, 'model_id']
best_model = h2o.get_model(best_model_id)

# Display the model summary to verify
best_model.summary()