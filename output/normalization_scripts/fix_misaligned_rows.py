import pandas as pd
import os

# Load the CSV file
csv_path = "output/cleaned_dataset/AGE_amharic_english.csv"
df = pd.read_csv(csv_path)

print("=" * 70)
print("FIXING MISALIGNED TRANSLATIONS (rows starting with '-')")
print("=" * 70)
print(f"\nInitial rows: {len(df)}")

# Find all rows where Amharic column starts with '-'
misaligned_indices = []
for idx in range(len(df) - 1):  # -1 to ensure we have a next row
    amharic = str(df.iloc[idx, 0]).strip()
    if amharic == "-":
        misaligned_indices.append(idx)

print(f"Found {len(misaligned_indices)} misaligned rows starting with '-'")

# Process misaligned rows (in reverse to avoid index shifting)
rows_to_drop = []
for idx in reversed(misaligned_indices):
    english_current = str(df.iloc[idx, 1]).strip()
    english_next = str(df.iloc[idx + 1, 1]).strip()

    # Merge: English from current row + English from next row
    merged_english = english_current + english_next

    # Update the next row with merged English
    df.iloc[idx + 1, 1] = merged_english

    # Mark current row for deletion
    rows_to_drop.append(idx)

    print(f"\n✓ Merged row {idx + 2} (header is row 1):")
    print(f"  Removed row {idx + 2} (had '-')")
    print(f"  Updated row {idx + 3}: English = '{english_current}...' + '{english_next}...'")

# Drop misaligned rows
df = df.drop(rows_to_drop).reset_index(drop=True)

print(f"\nFinal rows: {len(df)}")
print(f"Rows removed: {len(rows_to_drop)}")

# Save the corrected file
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved corrected file → {csv_path}")

# Verify: check if any '-' rows remain
remaining_dashes = (df.iloc[:, 0].str.strip() == "-").sum()
print(f"\nVerification: Remaining '-' rows: {remaining_dashes}")

if remaining_dashes == 0:
    print("✓ All misaligned rows fixed!")
else:
    print(f"⚠ Warning: {remaining_dashes} '-' rows still exist")

print("=" * 70)
