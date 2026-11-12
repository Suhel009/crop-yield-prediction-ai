import pandas as pd

df = pd.read_csv("dataset/crop_yield.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA INFO =====")
df.info()

print("\n===== STATISTICS =====")
print(df.describe())