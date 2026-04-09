import pandas as pd
output_file = "all_data.csv"

# read csv, shuffle the rows and save it, conserving both columns into training and eval files

data = pd.read_csv(output_file)
print(data.head())
data = data.sample(frac=1).reset_index(drop=True)  # shuffle the data
train_size = int(0.8 * len(data))
train_data = data[:train_size]
eval_data = data[train_size:]
train_data.to_csv("train.csv", index=False)
eval_data.to_csv("val.csv", index=False)