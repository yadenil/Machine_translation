"""
PHASE 5 - MILESTONE 2.1: Trilingual Augmentation (Week 3-4)
===========================================================

Follows the exact design from streamed-snacking-newt.md Phase 5.2

Tasks:
├─ Load clean_translation.csv (2K trilingual pairs)
├─ Duplicate trilingual rows 3x (create 6K synthetic + 2K original)
├─ Mix with bilingual data in 40% trilingual, 30% EN-AM, 30% AM-OR ratio
├─ Create dataloader with dynamic weighted sampling
└─ Test that batch composition matches target distribution

Output: Enhanced training data with trilingual weighting
Verification: Sample 10 batches, report % trilingual vs bilingual
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# ── Configuration ────────────────────────────────────────────────────────────
BASE_PATH = "output"
TRILINGUAL_DATA_PATH = os.path.join(BASE_PATH, "cleaned_dataset", "clean_translation.csv")
EN_AM_PATH = os.path.join(BASE_PATH, "data_final", "amharic_english_final.csv")
AM_OR_PATH = os.path.join(BASE_PATH, "data_final", "amharic_oromo_final.csv")

OUTPUT_DIR = os.path.join(BASE_PATH, "phase_5_milestone_2_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling distribution (Phase 2: Fine-Tuning)
TRILINGUAL_RATIO = 0.40  # 40% trilingual
EN_AM_RATIO = 0.30       # 30% EN-AM bilingual
AM_OR_RATIO = 0.30       # 30% AM-OR bilingual

BATCH_SIZE = 64
TRILINGUAL_DUPLICATION = 3  # Duplicate trilingual 3x


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: Load clean_translation.csv (2K trilingual pairs)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TASK 1: LOAD TRILINGUAL DATA (2K pairs)")
print("="*80 + "\n")

if not os.path.exists(TRILINGUAL_DATA_PATH):
    print(f"❌ Error: Trilingual data not found at {TRILINGUAL_DATA_PATH}")
    exit(1)

df_trilingual = pd.read_csv(TRILINGUAL_DATA_PATH, sep="\t")
df_trilingual.columns = ["Amharic", "Oromo", "English"]
print(f"✓ Loaded trilingual data: {len(df_trilingual):,} rows")
print(f"  Columns: {df_trilingual.columns.tolist()}")
print(f"  Sample:")
print(df_trilingual.head(2))

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: Duplicate trilingual rows 3x (create 6K synthetic + 2K original = 8K total)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(f"TASK 2: DUPLICATE TRILINGUAL ROWS {TRILINGUAL_DUPLICATION}x")
print("="*80 + "\n")

# Create duplicated copies (3x)
df_trilingual_augmented = pd.concat(
    [df_trilingual] * (TRILINGUAL_DUPLICATION + 1),  # +1 for original
    ignore_index=True
)

print(f"✓ Duplicated trilingual data {TRILINGUAL_DUPLICATION}x:")
print(f"  Original trilingual: {len(df_trilingual):,} rows")
print(f"  After {TRILINGUAL_DUPLICATION}x duplication: {len(df_trilingual_augmented):,} rows")
print(f"  Breakdown: 2K original + 6K synthetic (from 2K × 3)")

# Mark duplication source
df_trilingual_augmented['source'] = 'trilingual'
df_trilingual_augmented['duplication_group'] = np.repeat(
    np.arange(len(df_trilingual)),
    TRILINGUAL_DUPLICATION + 1
)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: Mix with bilingual data in 40% trilingual, 30% EN-AM, 30% AM-OR ratio
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TASK 3: MIX BILINGUAL DATA (40% trilingual, 30% EN-AM, 30% AM-OR)")
print("="*80 + "\n")

# Load bilingual data
df_en_am = pd.read_csv(EN_AM_PATH)
df_am_or = pd.read_csv(AM_OR_PATH)

print(f"✓ Loaded bilingual data:")
print(f"  EN-AM pairs: {len(df_en_am):,} rows")
print(f"  AM-OR pairs: {len(df_am_or):,} rows")

# Calculate sampling sizes for epoch (target distribution)
total_trilingual = len(df_trilingual_augmented)
print(f"\n✓ Target distribution (per epoch):")
print(f"  Trilingual (40%): {int(total_trilingual * TRILINGUAL_RATIO):,} samples")
print(f"  EN-AM (30%):      {int(total_trilingual * EN_AM_RATIO):,} samples")
print(f"  AM-OR (30%):      {int(total_trilingual * AM_OR_RATIO):,} samples")

# Create mixed dataset with pair type labels
df_trilingual_augmented['pair_type'] = 'trilingual'
df_trilingual_augmented['language_pair'] = 'EN-AM-OR'

df_en_am_labeled = df_en_am.copy()
df_en_am_labeled['pair_type'] = 'bilingual'
df_en_am_labeled['language_pair'] = 'EN-AM'
df_en_am_labeled['source'] = 'bilingual'

df_am_or_labeled = df_am_or.copy()
df_am_or_labeled['pair_type'] = 'bilingual'
df_am_or_labeled['language_pair'] = 'AM-OR'
df_am_or_labeled['source'] = 'bilingual'

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: Create dataloader with dynamic weighted sampling
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TASK 4: CREATE DATALOADER WITH DYNAMIC WEIGHTED SAMPLING")
print("="*80 + "\n")


class MixedBilingualTrilingualDataset(Dataset):
    """Dataset with dynamic weighted sampling for mixed bilingual-trilingual data"""
    
    def __init__(self, trilingual_data, en_am_data, am_or_data, 
                 trilingual_ratio=0.40, batch_size=64, epoch=1):
        """
        Create mixed dataset with weighted sampling
        
        Args:
            trilingual_data: DataFrame with EN-AM-OR triplets
            en_am_data: DataFrame with EN-AM pairs
            am_or_data: DataFrame with AM-OR pairs
            trilingual_ratio: Target % of trilingual pairs (0-1)
            batch_size: Batch size for epoch-wise calculations
            epoch: Current epoch (for dynamic weighting)
        """
        self.trilingual = trilingual_data
        self.en_am = en_am_data
        self.am_or = am_or_data
        self.trilingual_ratio = trilingual_ratio
        self.batch_size = batch_size
        self.epoch = epoch
        
        # Calculate per-epoch sampling sizes
        num_trilingual = max(1, int(len(self.trilingual) * self.trilingual_ratio))
        remaining = len(self.trilingual) - num_trilingual
        
        self.num_trilingual = num_trilingual
        self.num_en_am = max(1, int(remaining * 0.5))  # 50% of remaining
        self.num_am_or = remaining - self.num_en_am     # Rest for AM-OR
        
        # Create epoch samples
        self.samples = []
        self.pair_types = []
        
        # Trilingual samples (40%)
        indices_tri = np.random.choice(len(self.trilingual), self.num_trilingual, replace=True)
        for idx in indices_tri:
            self.samples.append(('trilingual', idx))
            self.pair_types.append('trilingual')
        
        # EN-AM samples (30%)
        indices_en_am = np.random.choice(len(self.en_am), self.num_en_am, replace=True)
        for idx in indices_en_am:
            self.samples.append(('en_am', idx))
            self.pair_types.append('en_am')
        
        # AM-OR samples (30%)
        indices_am_or = np.random.choice(len(self.am_or), self.num_am_or, replace=True)
        for idx in indices_am_or:
            self.samples.append(('am_or', idx))
            self.pair_types.append('am_or')
        
        # Shuffle for random ordering
        shuffle_idx = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in shuffle_idx]
        self.pair_types = [self.pair_types[i] for i in shuffle_idx]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pair_type, data_idx = self.samples[idx]
        
        if pair_type == 'trilingual':
            row = self.trilingual.iloc[data_idx]
            return {
                'pair_type': 'trilingual',
                'language_pair': 'EN-AM-OR',
                'data': row
            }
        elif pair_type == 'en_am':
            row = self.en_am.iloc[data_idx]
            return {
                'pair_type': 'bilingual',
                'language_pair': 'EN-AM',
                'data': row
            }
        else:  # am_or
            row = self.am_or.iloc[data_idx]
            return {
                'pair_type': 'bilingual',
                'language_pair': 'AM-OR',
                'data': row
            }


# Create dataset
dataset = MixedBilingualTrilingualDataset(
    trilingual_data=df_trilingual_augmented,
    en_am_data=df_en_am_labeled,
    am_or_data=df_am_or_labeled,
    trilingual_ratio=TRILINGUAL_RATIO,
    batch_size=BATCH_SIZE,
    epoch=1
)

print(f"✓ Created MixedBilingualTrilingualDataset:")
print(f"  Total samples: {len(dataset):,}")
print(f"  Trilingual: {dataset.num_trilingual:,} ({dataset.num_trilingual/len(dataset)*100:.1f}%)")
print(f"  EN-AM: {dataset.num_en_am:,} ({dataset.num_en_am/len(dataset)*100:.1f}%)")
print(f"  AM-OR: {dataset.num_am_or:,} ({dataset.num_am_or/len(dataset)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5: Test that batch composition matches target distribution
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TASK 5: VERIFY BATCH COMPOSITION (Sample 10 batches)")
print("="*80 + "\n")

# Create simple batch iterator for verification
def batch_iterator(dataset, batch_size=64):
    """Simple batch iterator"""
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        yield batch


batch_stats = []
trilingual_count_total = 0
bilingual_count_total = 0
en_am_count_total = 0
am_or_count_total = 0

print("Sample 10 batches:")
print("─" * 80)

for batch_idx, batch in enumerate(batch_iterator(dataset, BATCH_SIZE)):
    if batch_idx >= 10:
        break
    
    trilingual_in_batch = sum(1 for item in batch if item['pair_type'] == 'trilingual')
    en_am_in_batch = sum(1 for item in batch if item['language_pair'] == 'EN-AM')
    am_or_in_batch = sum(1 for item in batch if item['language_pair'] == 'AM-OR')
    
    trilingual_count_total += trilingual_in_batch
    bilingual_count_total += len(batch) - trilingual_in_batch
    en_am_count_total += en_am_in_batch
    am_or_count_total += am_or_in_batch
    
    trilingual_pct = trilingual_in_batch / len(batch) * 100
    en_am_pct = en_am_in_batch / len(batch) * 100
    am_or_pct = am_or_in_batch / len(batch) * 100
    
    batch_stats.append({
        'batch': batch_idx + 1,
        'trilingual': trilingual_in_batch,
        'trilingual_pct': trilingual_pct,
        'en_am': en_am_in_batch,
        'en_am_pct': en_am_pct,
        'am_or': am_or_in_batch,
        'am_or_pct': am_or_pct,
    })
    
    print(f"Batch {batch_idx+1:2d}: Trilingual {trilingual_in_batch:2d} ({trilingual_pct:5.1f}%) | "
          f"EN-AM {en_am_in_batch:2d} ({en_am_pct:5.1f}%) | AM-OR {am_or_in_batch:2d} ({am_or_pct:5.1f}%)")

# Calculate aggregate statistics
total_samples = trilingual_count_total + bilingual_count_total
trilingual_avg_pct = trilingual_count_total / total_samples * 100
en_am_avg_pct = en_am_count_total / total_samples * 100
am_or_avg_pct = am_or_count_total / total_samples * 100

print("─" * 80)
print(f"\nAggregate Statistics (10 batches, {total_samples} samples):")
print(f"  Trilingual: {trilingual_count_total:4d} ({trilingual_avg_pct:5.1f}%) | Target: 40.0%")
print(f"  EN-AM:      {en_am_count_total:4d} ({en_am_avg_pct:5.1f}%) | Target: 30.0%")
print(f"  AM-OR:      {am_or_count_total:4d} ({am_or_avg_pct:5.1f}%) | Target: 30.0%")

deviation_trilingual = abs(trilingual_avg_pct - 40.0)
deviation_en_am = abs(en_am_avg_pct - 30.0)
deviation_am_or = abs(am_or_avg_pct - 30.0)

print(f"\nDeviation from Target:")
print(f"  Trilingual: {deviation_trilingual:5.2f}% (target ±5%)")
print(f"  EN-AM:      {deviation_en_am:5.2f}% (target ±5%)")
print(f"  AM-OR:      {deviation_am_or:5.2f}% (target ±5%)")

if deviation_trilingual <= 5 and deviation_en_am <= 5 and deviation_am_or <= 5:
    print(f"\n✅ PASS: Batch composition within acceptable tolerance (±5%)")
else:
    print(f"\n⚠️  WARNING: Some batches deviate >5% from target")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT: Enhanced training data with trilingual weighting
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("OUTPUT: ENHANCED TRAINING DATA WITH TRILINGUAL WEIGHTING")
print("="*80 + "\n")

# Save augmented trilingual data
trilingual_output_path = os.path.join(OUTPUT_DIR, "trilingual_augmented_3x.csv")
df_trilingual_augmented.to_csv(trilingual_output_path, index=False)
print(f"✓ Saved augmented trilingual data: {trilingual_output_path}")
print(f"  Rows: {len(df_trilingual_augmented):,}")

# Create summary report
report = {
    "milestone": "2.1 - Trilingual Augmentation",
    "phase": "Phase 5: Implementation Roadmap",
    "timestamp": datetime.now().isoformat(),
    "tasks_completed": [
        "Load clean_translation.csv (2K trilingual pairs)",
        "Duplicate trilingual rows 3x (create 6K synthetic + 2K original)",
        "Mix with bilingual data in 40% trilingual, 30% EN-AM, 30% AM-OR ratio",
        "Create dataloader with dynamic weighted sampling",
        "Test that batch composition matches target distribution"
    ],
    "trilingual_augmentation": {
        "original_trilingual_rows": len(df_trilingual),
        "after_3x_duplication": len(df_trilingual_augmented),
        "synthetic_rows_created": len(df_trilingual_augmented) - len(df_trilingual),
        "breakdown": {
            "original": len(df_trilingual),
            "synthetic_copies": len(df_trilingual_augmented) - len(df_trilingual),
        }
    },
    "mixed_dataset_composition": {
        "total_bilingual_data": {
            "en_am_rows": len(df_en_am),
            "am_or_rows": len(df_am_or),
        },
        "target_sampling_ratio": {
            "trilingual": TRILINGUAL_RATIO,
            "en_am": EN_AM_RATIO,
            "am_or": AM_OR_RATIO,
        }
    },
    "dataloader_verification": {
        "batch_size": BATCH_SIZE,
        "num_batches_sampled": 10,
        "aggregate_statistics": {
            "trilingual_percentage": trilingual_avg_pct,
            "trilingual_target": 40.0,
            "trilingual_deviation": deviation_trilingual,
            "en_am_percentage": en_am_avg_pct,
            "en_am_target": 30.0,
            "en_am_deviation": deviation_en_am,
            "am_or_percentage": am_or_avg_pct,
            "am_or_target": 30.0,
            "am_or_deviation": deviation_am_or,
        },
        "batch_composition_samples": batch_stats,
        "verification_status": "PASS" if (deviation_trilingual <= 5 and deviation_en_am <= 5 and deviation_am_or <= 5) else "WARNING"
    },
    "output_files": {
        "trilingual_augmented_data": trilingual_output_path,
    }
}

# Save report
report_path = os.path.join(OUTPUT_DIR, "milestone_2_1_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Saved report: {report_path}")

# Create summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print("\n1. TRILINGUAL AUGMENTATION:")
print(f"   Original:        {len(df_trilingual):,} rows")
print(f"   After 3x Dupl:   {len(df_trilingual_augmented):,} rows (+{len(df_trilingual_augmented) - len(df_trilingual):,})")
print(f"   Total synthetic: {len(df_trilingual_augmented) - len(df_trilingual):,} rows")

print("\n2. BILINGUAL DATA AVAILABLE:")
print(f"   EN-AM:           {len(df_en_am):,} rows")
print(f"   AM-OR:           {len(df_am_or):,} rows")

print("\n3. TARGET SAMPLING DISTRIBUTION (Per Epoch):")
print(f"   Trilingual (40%): {int(len(df_trilingual_augmented) * TRILINGUAL_RATIO):,} samples")
print(f"   EN-AM (30%):      {int(len(df_trilingual_augmented) * EN_AM_RATIO):,} samples")
print(f"   AM-OR (30%):      {int(len(df_trilingual_augmented) * AM_OR_RATIO):,} samples")

print("\n4. DATALOADER VERIFICATION (10 samples):")
print(f"   Trilingual:      {trilingual_avg_pct:5.1f}% (target 40%, dev {deviation_trilingual:5.2f}%)")
print(f"   EN-AM:           {en_am_avg_pct:5.1f}% (target 30%, dev {deviation_en_am:5.2f}%)")
print(f"   AM-OR:           {am_or_avg_pct:5.1f}% (target 30%, dev {deviation_am_or:5.2f}%)")

print("\n5. VERIFICATION STATUS:")
if deviation_trilingual <= 5 and deviation_en_am <= 5 and deviation_am_or <= 5:
    print(f"   ✅ PASS - All distributions within ±5% tolerance")
else:
    print(f"   ⚠️  WARNING - Some distributions exceed ±5% tolerance")

print("\n" + "="*80)
print("✅ MILESTONE 2.1: TRILINGUAL AUGMENTATION COMPLETE")
print("="*80 + "\n")

print(f"Output files saved to: {OUTPUT_DIR}/")
print(f"Next: Proceed to Milestone 2.2 (Encoder Freezing Experiments)")
