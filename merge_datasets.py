import pandas as pd
import os

print("=" * 50)
print("STEP 1: Loading your existing dataset...")
print("=" * 50)

spotify_df = pd.read_csv('final_dataset.csv')
print(f"✅ Spotify dataset loaded: {spotify_df.shape[0]} songs")

print("\nSTEP 2: Loading Indian songs dataset...")
indian_df = pd.read_csv('indian_songs.csv')
print(f"✅ Indian songs loaded: {indian_df.shape[0]} songs")

if 'Track ID' not in indian_df.columns:
    start_id = int(spotify_df['Track ID'].max()) + 1 if 'Track ID' in spotify_df.columns else 7001
    indian_df.insert(0, 'Track ID', range(start_id, start_id + len(indian_df)))

print("\nSTEP 3: Merging datasets...")
merged_df = pd.concat([spotify_df, indian_df], ignore_index=True)
merged_df = merged_df.drop_duplicates(subset=['track_name', 'artists'])
merged_df = merged_df.reset_index(drop=True)
print(f"✅ Merged dataset: {merged_df.shape[0]} total songs")

merged_df.to_csv('dataset.csv', index=False)
print("\n✅ Saved as dataset.csv")
print("=" * 50)
print("ALL DONE! Now run: streamlit run app.py")
print("=" * 50)
