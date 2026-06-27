# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("Titanic-Dataset.csv")

# First 5 Rows
print(data.head())

# Dataset Information
print(data.info())

# Missing Values
print(data.isnull().sum())

# Fill Missing Age Values
data["Age"] = data["Age"].fillna(data["Age"].mean())

# ---------------------------
# Survival Count
# ---------------------------
plt.figure(figsize=(6,4))
data["Survived"].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()

# ---------------------------
# Gender Distribution
# ---------------------------
plt.figure(figsize=(6,4))
data["Sex"].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ---------------------------
# Age Distribution
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(data["Age"], bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# Passenger Class Distribution
# ---------------------------
plt.figure(figsize=(6,4))
data["Pclass"].value_counts().sort_index().plot(kind='bar')
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ---------------------------
# Survival by Gender
# ---------------------------
survival_gender = pd.crosstab(data['Sex'], data['Survived'])

survival_gender.plot(kind='bar', figsize=(6,4))
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ---------------------------
# Survival by Passenger Class
# ---------------------------
survival_class = pd.crosstab(data['Pclass'], data['Survived'])

survival_class.plot(kind='bar', figsize=(6,4))
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()