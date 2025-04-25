import pandas as pd

# Read the original CSV file into a DataFrame.
df = pd.read_csv("input_file.csv")  # Replace with your CSV file name

# Filter the DataFrame for rows where the state is "NY".
filtered_df = df[df["State"] == "NY"]

# Write the filtered DataFrame to a new CSV file.
filtered_df.to_csv("filtered_output.csv", index=False)  # Replace with your desired output file name

print("Filtered CSV file with state 'NY' has been successfully created.")
