import pandas as pd

# Load the existing dataset
data = pd.read_csv('data/dataset.csv')

# Add a new feature
data['feature3'] = data['feature1'] + data['feature2']

# Save the modified dataset
data.to_csv('data/dataset.csv', index=False)
