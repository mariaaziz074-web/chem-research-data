"""
Basic validation example (Python script version)
No Jupyter needed!
"""

import pandas as pd
from chemdata.validation import ChemicalValidator, NumericalValidator

# Create sample data
data = {
    'catalyst': ['TiO2', 'ZnO', 'Fe2O3', 'CuO'],
    'concentration_mg_L': [10.0, 20.0, 15.0, 25.0],
    'pH': [7.0, 6.5, 8.0, 7.5],
    'temperature_C': [25.0, 30.0, 25.0, 35.0],
    'degradation_percent': [85.0, 92.0, 78.0, 88.0],
}
df = pd.DataFrame(data)
print("Sample Data:")
print(df)
print("\n" + "="*60 + "\n")

# Test 1: Validate formulas
print("TEST 1: Chemical Formula Validation")
print("-" * 40)
chem_validator = ChemicalValidator()

for formula in df['catalyst']:
    result = chem_validator.validate_formula(formula)
    print(f"{formula:10} → Valid: {result['valid']:5} | MW: {result['molecular_weight']:.2f} g/mol")

print("\n" + "="*60 + "\n")

# Test 2: Validate numerical ranges
print("TEST 2: Numerical Validation")
print("-" * 40)
num_validator = NumericalValidator()

# Validate pH
for i, ph in enumerate(df['pH']):
    result = num_validator.validate_ph(ph)
    status = "✓" if result['valid'] else "✗"
    print(f"Row {i}: pH = {ph:4.1f} {status}")

print("\n" + "="*60 + "\n")

# Test 3: Validate entire dataframe
print("TEST 3: DataFrame Column Validation")
print("-" * 40)

df_validated = num_validator.validate_column(
    df,
    'degradation_percent',
    validator_type='percentage'
)

print(df_validated[['degradation_percent', 'degradation_percent_valid']])

print("\n" + "="*60)
print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)