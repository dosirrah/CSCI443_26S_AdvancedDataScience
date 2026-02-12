# lecture03/example1_data_frames.py
import pandas as pd

# Data to be represented in the DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [20, 21, 19, 22, 20],
    'Grade': [88, 92, 85, 90, 95]
}

# Create a DataFrame from the data
students_df = pd.DataFrame(data)

# Display the DataFrame
print(students_df)
