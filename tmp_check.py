import pandas as pd
df = pd.read_parquet(r"data\02_intermediate\mt_patients_clean.parquet")
print(df.shape, df.columns.tolist()[:20])
print(df.dtypes.head(10))
