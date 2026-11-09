import pandas as pd

# Check CDMs
df = pd.read_csv(r"E:\DL\data\cdms.csv")
print("=== CDM Data ===")
print(f"Total conjunctions: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample rows:")
print(df[['SAT_1_NAME', 'SAT_2_NAME', 'MIN_RNG', 'PC', 'TCA']].head(5))
print(f"\nMiss distance stats (km):")
print(df['MIN_RNG'].describe())
print(f"\nCollision probability stats:")
print(df['PC'].describe())