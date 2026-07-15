# Extracted from notebook

# %% Cell 1
This project analyzes customer support data from Flipkart to evaluate customer satisfaction (CSAT), agent performance, complaint categories, product categories, response times, and customer interactions. The goal is to discover patterns that help improve customer service quality and operational efficiency using Exploratory Data Analysis (EDA).

# %% Cell 2
#Import required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# %% Cell 3
#load dataset
file_path = "C:/Users/yashw/Downloads/Flipcart Project/Customer_support_data.csv"
df = pd.read_csv(file_path)

# %% Cell 4

#dataset firstview
print("Rows and Columns:", df.shape)

df.info()
# First 5 rows
df.head()
# Last 5 rows
df.tail()

# %% Cell 5
# Dataset Rows & Columns

rows, columns = df.shape

print("Number of Rows    :", rows)
print("Number of Columns :", columns)

# %% Cell 6
# Dataset Info# Dataset Information

df.info()

# %% Cell 7
## Check duplicate values

duplicate_count = df.duplicated().sum()

print("Number of Duplicate Rows:", duplicate_count)

# %% Cell 8
# Missing Values/Null Values Count
# Missing Values

df.isnull().sum()

# %% Cell 9
# Visualizing Missing Values

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Records")
plt.show()

# %% Cell 10
# Dataset Columns
# Dataset Columns

df.columns
# Dataset Columns

for col in df.columns:
    print(col)

# %% Cell 11
# Dataset De# Statistical Summary of Numerical Columns

df.describe()

# %% Cell 12
# Check Unique Values for Each Variable

for column in df.columns:
    print(f"\nColumn Name : {column}")
    print(f"Number of Unique Values : {df[column].nunique()}")
    print(df[column].unique())
    print("-" * 80)# Check Unique Values for each variable.

# %% Cell 13
# DATA WRANGLING

# Make a copy of the dataset
df1 = df.copy()

# Check duplicate records
print("Duplicate Rows:", df1.duplicated().sum())

# Remove duplicate rows
df1.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:")
print(df1.isnull().sum())

# Convert date columns to datetime format
date_columns = [
    'order_date_time',
    'Issue_reported at',
    'issue_responded',
    'Survey_response_Date'
]

for col in date_columns:
    df1[col] = pd.to_datetime(df1[col], errors='coerce')

# Fill missing values

# Numerical columns
df1['Item_price'].fillna(df1['Item_price'].median(), inplace=True)
df1['connected_handling_time'].fillna(df1['connected_handling_time'].median(), inplace=True)

# Categorical columns
categorical_columns = [
    'channel_name',
    'category',
    'Sub-category',
    'Customer_City',
    'Product_category',
    'Agent_name',
    'Supervisor',
    'Manager',
    'Tenure Bucket',
    'Agent Shift'
]

for col in categorical_columns:
    df1[col].fillna(df1[col].mode()[0], inplace=True)

# Create Response Time (in minutes)
df1['response_time'] = (
    df1['issue_responded'] - df1['Issue_reported at']
).dt.total_seconds() / 60

# Fill missing response time
df1['response_time'].fillna(df1['response_time'].median(), inplace=True)

# Check final dataset
print("\nFinal Shape:", df1.shape)

df1.head()

# %% Cell 14
# Chart - 1 visualization code
plt.figure(figsize=(8,5))
sns.countplot(data=df1, x='CSAT Score', palette='viridis')
plt.title("Distribution of CSAT Scores")
plt.xlabel("CSAT Score")
plt.ylabel("Count")
plt.show()

# %% Cell 15
# Chart - 2 visualization code
plt.figure(figsize=(8,5))
sns.countplot(data=df1, x='channel_name', palette='Set2')
plt.title("Support Channel Distribution")
plt.xticks(rotation=45)
plt.show()

# %% Cell 16
# Chart - 3 visualization code
plt.figure(figsize=(10,5))
df1['category'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Complaint Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# %% Cell 17
# Chart - 4 visualization code
plt.figure(figsize=(12,6))
df1['Sub-category'].value_counts().head(10).plot(kind='barh', color='orange')
plt.title("Top 10 Complaint Sub-Categories")
plt.xlabel("Count")
plt.show()

# %% Cell 18
# Chart - 5 visualization code
plt.figure(figsize=(12,6))
df1['Product_category'].value_counts().plot(kind='bar', color='green')
plt.title("Product Category Distribution")
plt.xticks(rotation=45)
plt.show()

# %% Cell 19
# Chart - 6 visualization code
plt.figure(figsize=(12,6))
df1['Customer_City'].value_counts().head(10).plot(kind='barh', color='purple')
plt.title("Top 10 Customer Cities")
plt.xlabel("Number of Tickets")
plt.show()

# %% Cell 20
# Chart - 7 visualization code
plt.figure(figsize=(12,6))

agent = df1.groupby('Agent_name')['CSAT Score'].mean().sort_values(ascending=False)

agent.head(10).plot(kind='bar', color='red')

plt.title("Top 10 Agents by Average CSAT")
plt.ylabel("Average CSAT")
plt.show()

# %% Cell 21
# Chart - 8 visualization code
plt.figure(figsize=(12,6))

supervisor = df1.groupby('Supervisor')['CSAT Score'].mean()

supervisor.plot(kind='bar', color='teal')

plt.title("Supervisor Performance")
plt.ylabel("Average CSAT")
plt.show()

# %% Cell 22
# Chart - 9 visualization code
plt.figure(figsize=(8,5))

manager = df1.groupby('Manager')['CSAT Score'].mean()

manager.plot(kind='bar', color='brown')

plt.title("Manager Performance")
plt.ylabel("Average CSAT")
plt.show()

# %% Cell 23
# Chart - 10 visualization code
plt.figure(figsize=(8,5))

sns.countplot(data=df1, x='Agent Shift', palette='Set3')

plt.title("Agent Shift Distribution")

plt.show()

# %% Cell 24
# Chart - 11 visualization code
plt.figure(figsize=(10,5))

sns.histplot(df1['Item_price'], bins=30, kde=True)

plt.title("Item Price Distribution")

plt.xlabel("Item Price")

plt.show()

# %% Cell 25
# Chart - 12 visualization code
df1['Month'] = df1['order_date_time'].dt.to_period('M')

monthly = df1.groupby('Month').size()

plt.figure(figsize=(12,6))

monthly.plot(marker='o')

plt.title("Monthly Complaint Trend")

plt.ylabel("Tickets")

plt.show()

# %% Cell 26
# Chart - 13 visualization code
plt.figure(figsize=(10,5))

sns.histplot(df1['response_time'], bins=30, kde=True, color='steelblue')

plt.title("Response Time Distribution")
plt.xlabel("Response Time (Minutes)")
plt.ylabel("Frequency")

plt.show()

# %% Cell 27
# Correlation Heatmap visualization code
plt.figure(figsize=(8,6))

numeric = df1.select_dtypes(include=['int64','float64'])

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.show()

# %% Cell 28
# Pair Plot visualization code
sample_df = df.sample(n=1000, random_state=42)

numeric = sample_df.select_dtypes(include=['int64','float64'])

sns.pairplot(numeric)

plt.show()

# %% Cell 29
print("EDA Capstone Project Completed Successfully!")

# %% Cell 30
print("Thank You!")

