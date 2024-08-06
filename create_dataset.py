# create_dataset.py
import numpy as np
import pandas as pd

# Create a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.choice([0, 1], 100)
})

# Save the dataset
data.to_csv('data/dataset.csv', index=False)