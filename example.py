# %%
import pandas as pd

# Load data (keep all columns)
df = pd.read_csv('zt_logs.csv')

# Ensure timestamp exists: create from date+time if necessary, otherwise parse existing timestamp
if 'timestamp' not in df.columns and {'date', 'time'}.issubset(df.columns):
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Sort by user then timestamp (keeps all columns)
df = df.sort_values(by=['user', 'timestamp'], ascending=[True, True]).reset_index(drop=True)

# Compute time diff within each user (seconds and minutes)
df['time_diff_seconds'] = df.groupby('user')['timestamp'].diff().dt.total_seconds()
df['time_diff_minutes'] = df['time_diff_seconds'] / 60

# Get previous country for same user to detect change
df['prev_country'] = df.groupby('user')['country'].shift(1)

# Anomaly rule: time_diff_minutes < 180 AND country changed (prev_country != country)
df['is_anomaly'] = (
    df['time_diff_minutes'].notna()  # exclude first events which are NaN
    & (df['time_diff_minutes'] < 180)
    & (df['country'] != df['prev_country'])
).astype(int)

# The prev_country helper column is optional; drop it if you don't want it saved:
df = df.drop(columns=['prev_country'])

df.head(15) 

# Save result (keeps all original columns + new ones appended)
df.to_csv('zt_logs_with_time_diff_and_anomaly.csv', index=False)



# Quick preview
print(df[['user', 'timestamp', 'time_diff_minutes', 'country', 'is_anomaly']].head(15))
