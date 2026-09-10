"""
Complete workflow example (Python script version)
"""

import pandas as pd
from chemdata.validation import ChemicalValidator
from chemdata.normalization import UnitNormalizer, ChemicalNormalizer
from chemdata.quality import OutlierDetector
from chemdata.provenance import ProvenanceTracker

print("="*60)
print("COMPLETE WORKFLOW TEST")
print("="*60 + "\n")

# Sample data
data = {
    'catalyst': ['TiO2', 'Ti O2', 'ZnO', 'Fe2O3'],
    'concentration_mg_L': [10.0, 20.0, 15.0, 25.0],
    'pH': [7.0, 6.5, 8.0, 7.5],
    'temperature_C': [25.0, 30.0, 25.0, 35.0],
    'degradation_percent': [85.0, 92.0, 78.0, 88.0],
}
df = pd.DataFrame(data)

print("STEP 1: Original Data")
print("-" * 40)
print(df)
print()

# Step 2: Add provenance
print("STEP 2: Add Provenance Tracking")
print("-" * 40)
tracker = ProvenanceTracker()
df = tracker.add_provenance(
    df,
    source="Example Dataset",
    source_type="synthetic"
)
print(f"✓ Provenance added: {df['provenance_source'].iloc[0]}")
print()

# Step 3: Normalize chemistry
print("STEP 3: Normalize Chemical Formulas")
print("-" * 40)
chem_normalizer = ChemicalNormalizer()
df = chem_normalizer.normalize_dataframe_column(
    df,
    'catalyst',
    normalization_type='formula'
)
print("Before → After:")
for orig, norm in zip(data['catalyst'], df['catalyst_normalized']):
    print(f"  {orig:10} → {norm}")
print()

# Step 4: Unit conversion
print("STEP 4: Convert Temperature Units")
print("-" * 40)
unit_normalizer = UnitNormalizer()
df = unit_normalizer.normalize_column(
    df,
    'temperature_C',
    from_unit='C',
    to_unit='K',
    conversion_type='temperature'
)
print("°C → K:")
for c, k in zip(df['temperature_C'], df['temperature_C_K']):
    print(f"  {c:5.1f}°C → {k:6.2f}K")
print()

# Step 5: Outlier detection
print("STEP 5: Detect Outliers")
print("-" * 40)
outlier_detector = OutlierDetector()
df = outlier_detector.mark_outliers(
    df,
    'degradation_percent',
    methods=['zscore']
)
print("Degradation % | Outlier?")
for deg, outlier in zip(df['degradation_percent'], df['degradation_percent_outlier_zscore']):
    status = "Yes ⚠" if outlier else "No"
    print(f"  {deg:5.1f}%      | {status}")
print()

print("="*60)
print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")