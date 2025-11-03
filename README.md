# NCSA-Brazil
MSOE AI Club Research Group &amp; UIUC's NCSA Collaboration

## Deforestation to Economic Impact Analysis

This project analyzes how deforestation in the Brazilian Amazon connects to economic outcomes through agricultural production.

### Scripts

1. **NodeWork.py** - Network analysis of deforestation spread patterns using grid-based approach
2. **economic_impact_analysis.py** - Site-specific case studies linking deforestation to economic indicators

### Running the Economic Impact Analysis

```bash
# Make sure you have the virtual environment activated
cd NCSA-Brazil
python economic_impact_analysis.py
```

This will analyze 3 farm sites and 3 pasture sites, generating:
- Annual snapshot images showing land transformation (2000-2023)
- Time series plots correlating deforestation with economic indicators
- CSV files with yearly metrics
- Summary reports for each site

### Output Structure

```
output_economic_analysis/
├── farm_site_1/
│   ├── snapshots/
│   │   ├── 2000_snapshot.png
│   │   ├── 2003_snapshot.png
│   │   └── ... (yearly)
│   ├── timeseries_plot.png
│   ├── metrics_timeseries.csv
│   └── site_summary.txt
├── pasture_site_1/
└── combined_summary.txt
```

### Datasets Used

- **UMD Hansen Global Forest Change** - Forest loss detection (2000-2024)
- **MODIS MOD13Q1** - NDVI vegetation health indices
- **NOAA VIIRS Nighttime Lights** - Economic activity proxy
- **Landsat imagery** - RGB visualizations

### Requirements

See `requirements.txt` for all dependencies. Key packages:
- earthengine-api
- geemap
- pandas
- matplotlib
- numpy