import pandas as pd
import os

# Get current directory and set correct paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths using os.path.join
training_xlsx = os.path.join(current_dir, "Training_Set.xlsx")
testing_xlsx = os.path.join(current_dir, "Testing_Set.xlsx")
training_csv = os.path.join(current_dir, "train.csv")
testing_csv = os.path.join(current_dir, "test.csv")

try:
    # Convert Training Data
    if os.path.exists(training_xlsx):
        df_train = pd.read_excel(training_xlsx)
        df_train.to_csv(training_csv, index=False)
        print(f"Training data converted to: {training_csv}")
    else:
        print(f"Training file not found at: {training_xlsx}")

    # Convert Testing Data
    if os.path.exists(testing_xlsx):
        df_test = pd.read_excel(testing_xlsx)
        df_test.to_csv(testing_csv, index=False)
        print(f"Testing data converted to: {testing_csv}")
    else:
        print(f"Testing file not found at: {testing_xlsx}")

except Exception as e:
    print(f"Error: {str(e)}")
