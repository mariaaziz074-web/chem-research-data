# chemdata: Chemistry Dataset Quality Assurance Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for validating, normalizing, and quality-checking chemistry datasets for machine learning and computational research.

## Features

✅ **Chemical Validation**
- SMILES validation and canonicalization
- Molecular formula validation
- Molecular weight calculation

✅ **Numerical Validation**
- Range checking (pH, temperature, concentration)
- Unit consistency verification
- Type validation

✅ **Unit Normalization**
- Temperature (°C ↔ K ↔ °F)
- Energy (eV ↔ kcal/mol ↔ kJ/mol)
- Time (s ↔ min ↔ h)
- Concentration (mg/L ↔ mM ↔ ppm)

✅ **Quality Assessment**
- Duplicate detection (chemical and experimental)
- Outlier detection (Z-score, IQR, Isolation Forest)
- Missing data analysis

✅ **Provenance Tracking**
- Source tracking (DOI, citation, file)
- Transformation logging
- Quality flagging

✅ **Export Formats**
- CSV, JSON, Parquet
- Train/test splits
- ML-ready formats

## Installation

```bash
# Clone repository
git clone (https://github.com/mariaaziz074-web/chem-research-data)
cd chem-research-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .
