import pandas as pd

# Load EV dataset
df = pd.read_csv("dataset/india_ev_range_dataset.csv")

print("EV DATASET ANALYSIS")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nTarget Variable:")
print(df["real_world_range_km"].describe())

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution of EV Range
plt.figure(figsize=(8, 5))
sns.histplot(df["real_world_range_km"], bins=30, kde=True)
plt.title("Distribution of EV Range")
plt.xlabel("Real World Range (km)")
plt.ylabel("Count")
plt.show()

# 2. Battery Capacity vs EV Range
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="battery_capacity_kWh",
    y="real_world_range_km"
)
plt.title("Battery Capacity vs EV Range")
plt.show()

# 3. EV Range by Manufacturer
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="make",
    y="real_world_range_km"
)
plt.title("EV Range by Manufacturer")
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: EV Range Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["real_world_range_km"], bins=30, kde=True)
plt.title("Distribution of EV Range")
plt.xlabel("Real World Range (km)")
plt.show()

# Graph 2: Battery Capacity vs Range
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="battery_capacity_kWh",
    y="real_world_range_km"
)
plt.title("Battery Capacity vs EV Range")
plt.show()

# Graph 3: Range by Manufacturer
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="make", y="real_world_range_km")
plt.title("EV Range by Manufacturer")
plt.xticks(rotation=45)
plt.show()