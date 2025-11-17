import pandas as pd
df = pd.read_parquet("spikes_parquet/batch=00000.parquet", columns=["spikes"])
lengths = df["spikes"].apply(len)
print("rows=", len(df), "TN (unique)=", lengths.unique())
