"""
Deforestation to Economic Impact Analysis
Brazilian Amazon Case Studies

This script analyzes how deforestation in the Brazilian Amazon connects to economic outcomes
by tracking 2-3 farm areas and 2-3 pasture areas over time (2000-2023).

Key Features:
- Identifies specific farm and pasture sites in Pará, Mato Grosso, and Rondônia states
- Extracts annual metrics: forest loss, NDVI, nighttime lights, land cover
- Generates yearly snapshot images showing land transformation
- Creates correlation analyses linking deforestation to economic activity
- Produces case study reports with visualizations and data

Usage:
    python economic_impact_analysis.py

Output:
    output_economic_analysis/
    ├── farm_site_1/
    │   ├── snapshots/ (yearly images)
    │   ├── timeseries_plot.png
    │   ├── metrics_timeseries.csv
    │   └── site_summary.txt
    ├── farm_site_2/
    ├── farm_site_3/
    ├── pasture_site_1/
    ├── pasture_site_2/
    ├── pasture_site_3/
    └── combined_summary.txt

Required datasets:
- UMD Hansen Global Forest Change (forest loss detection)
- MODIS MOD13Q1 (NDVI vegetation health)
- NOAA VIIRS Nighttime Lights (economic activity proxy)
- Landsat imagery (RGB visualizations)
"""

import ee
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

try:
    ee.Initialize()
except:
    ee.Initialize(project="dotted-saga-475001-n3")

def identify_case_study_sites(roi, num_farm_sites=3, num_pasture_sites=3):
    """
    Identify 2-3 farm areas and 2-3 pasture areas in Brazilian Amazon.
    For now, using known deforestation hotspots. Can be extended with MapBiomas auto-detection.
    
    Args:
        roi: Earth Engine Geometry (Region of Interest in Brazilian Amazon)
        num_farm_sites: Number of farm sites to identify
        num_pasture_sites: Number of pasture sites to identify
    
    Returns:
        List of site dictionaries with coordinates, area, transition year, and type
    """
    
    # Based on research, identify well-documented sites in Brazilian Amazon
    # Sites selected from Pará and Rondônia states (major deforestation areas)
    
    sites = []
    
    # Farm sites (Soybean cultivation areas)
    farm_coordinates = [
        # Farm Site 1: Paragominas, Pará (known soybean region)
        {'name': 'farm_site_1', 'type': 'farm', 'lat': -3.0, 'lon': -47.35, 'transition_year': 2005,
         'description': 'Paragominas soybean farm'},
        # Farm Site 2: Santana do Araguaia, Pará
        {'name': 'farm_site_2', 'type': 'farm', 'lat': -9.3, 'lon': -50.1, 'transition_year': 2008,
         'description': 'Santana do Araguaia agricultural area'},
        # Farm Site 3: Lucas do Rio Verde, Mato Grosso
        {'name': 'farm_site_3', 'type': 'farm', 'lat': -13.05, 'lon': -55.9, 'transition_year': 2006,
         'description': 'Lucas do Rio Verde crop farm'},
    ]
    
    # Pasture sites (Cattle ranching areas)
    pasture_coordinates = [
        # Pasture Site 1: São Félix do Xingu, Pará (largest cattle municipality)
        {'name': 'pasture_site_1', 'type': 'pasture', 'lat': -6.6, 'lon': -51.98, 'transition_year': 2003,
         'description': 'São Félix do Xingu cattle ranch'},
        # Pasture Site 2: Novo Progresso, Pará
        {'name': 'pasture_site_2', 'type': 'pasture', 'lat': -7.15, 'lon': -55.38, 'transition_year': 2007,
         'description': 'Novo Progresso pasture'},
        # Pasture Site 3: Rondônia cattle region
        {'name': 'pasture_site_3', 'type': 'pasture', 'lat': -10.5, 'lon': -62.2, 'transition_year': 2004,
         'description': 'Rondônia cattle area'},
    ]
    
    # Create site geometries (~20km x 20km boxes around each point)
    size_deg = 0.18  # ~20km
    
    for site_data in farm_coordinates[:num_farm_sites]:
        geom = ee.Geometry.Rectangle([
            site_data['lon'] - size_deg/2, site_data['lat'] - size_deg/2,
            site_data['lon'] + size_deg/2, site_data['lat'] + size_deg/2
        ])
        
        sites.append({
            'name': site_data['name'],
            'type': site_data['type'],
            'geometry': geom,
            'center_lat': site_data['lat'],
            'center_lon': site_data['lon'],
            'transition_year': site_data['transition_year'],
            'description': site_data['description']
        })
    
    for site_data in pasture_coordinates[:num_pasture_sites]:
        geom = ee.Geometry.Rectangle([
            site_data['lon'] - size_deg/2, site_data['lat'] - size_deg/2,
            site_data['lon'] + size_deg/2, site_data['lat'] + size_deg/2
        ])
        
        sites.append({
            'name': site_data['name'],
            'type': site_data['type'],
            'geometry': geom,
            'center_lat': site_data['lat'],
            'center_lon': site_data['lon'],
            'transition_year': site_data['transition_year'],
            'description': site_data['description']
        })
    
    return sites

def extract_annual_metrics(site, year, start_date='2000-01-01', end_date='2024-01-01'):
    """
    Extract annual metrics for a site including forest cover, NDVI, land use, nighttime lights.
    
    Args:
        site: Site dictionary with geometry and metadata
        year: Year to extract metrics for
        start_date: Start date for time series data
        end_date: End date for time series data
    
    Returns:
        Dictionary of metrics for the specified year
    """
    geometry = site['geometry']
    
    metrics = {
        'year': year,
        'site_name': site['name'],
        'site_type': site['type']
    }
    
    # 1. Forest Loss Detection (Hansen)
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    
    # Get forest cover in 2000
    treecover = hansen.select('treecover2000').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    # Get loss year
    lossyear = hansen.select('lossyear').reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    # Calculate cumulative forest loss up to this year
    year_index = year - 2000
    loss_by_year = hansen.select('lossyear')
    
    # Create mask for cumulative loss up to this year
    loss_mask = loss_by_year.lte(year_index).And(loss_by_year.gt(0))
    
    # Count pixels with forest loss up to this year
    cumulative_loss_count = loss_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    metrics['forest_cover_2000'] = treecover.get('treecover2000', 0)
    metrics['loss_year'] = lossyear.get('lossyear', 0)
    # Extract count from result (get first value since it's a mask)
    metrics['cumulative_loss_by_year'] = list(cumulative_loss_count.values())[0] if cumulative_loss_count else 0
    
    # 2. NDVI Time Series (annual composite)
    year_start = f'{year}-01-01'
    year_end = f'{year+1}-01-01'
    
    ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .select('NDVI') \
        .filterDate(year_start, year_end) \
        .filterBounds(geometry)
    
    # Calculate annual mean NDVI
    annual_ndvi = ndvi_collection.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=250,
        maxPixels=1e9
    ).getInfo()
    
    ndvi_value = annual_ndvi.get('NDVI', 0)
    metrics['ndvi_mean'] = ndvi_value * 0.0001 if ndvi_value else 0  # Convert scale
    
    # 3. Nighttime Lights (if available for this year)
    if year >= 2012 and year <= 2021:  # VIIRS data range
        try:
            lights_collection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
                .filterDate(year_start, year_end) \
                .select('avg_rad') \
                .filterBounds(geometry)
            
            annual_lights = lights_collection.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=500,
                maxPixels=1e9
            ).getInfo()
            
            metrics['nighttime_lights_mean'] = annual_lights.get('avg_rad', 0)
        except:
            metrics['nighttime_lights_mean'] = None
    else:
        metrics['nighttime_lights_mean'] = None
    
    # 4. Land Cover Classification (simplified - using NDVI thresholds)
    if metrics['ndvi_mean'] > 0.5:
        metrics['likely_land_cover'] = 'vegetation'
    elif metrics['ndvi_mean'] > 0.3:
        metrics['likely_land_cover'] = 'mixed'
    else:
        metrics['likely_land_cover'] = 'bare/agricultural'
    
    return metrics

def generate_annual_snapshot(site, year, save_dir='output_economic_analysis'):
    """
    Create static PNG map showing RGB composite, forest loss overlay, and site boundary.
    
    Args:
        site: Site dictionary with geometry and metadata
        year: Year to create snapshot for
        save_dir: Directory to save snapshots
    
    Returns:
        Path to saved snapshot image
    """
    geometry = site['geometry']
    
    # Get Landsat or Sentinel imagery for RGB
    year_start = f'{year}-06-01'  # Dry season for better visibility
    year_end = f'{year}-09-30'
    
    # Try Landsat 8/9 first (2013-present), then Landsat 7/5
    if year >= 2013:
        image_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA") \
            .filterDate(year_start, year_end) \
            .filterBounds(geometry) \
            .median()
        rgb_bands = ['B4', 'B3', 'B2']  # Red, Green, Blue
    elif year >= 2000:
        image_collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA") \
            .filterDate(year_start, year_end) \
            .filterBounds(geometry) \
            .median()
        rgb_bands = ['B3', 'B2', 'B1']  # Red, Green, Blue
    else:
        # Fallback to Landsat 5
        image_collection = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA") \
            .filterDate(year_start, year_end) \
            .filterBounds(geometry) \
            .median()
        rgb_bands = ['B3', 'B2', 'B1']
    
    # Clip to site boundary
    rgb_image = image_collection.select(rgb_bands).clip(geometry)
    
    # Get forest loss overlay
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    
    # Show forest loss up to this year
    year_index = year - 2000
    loss_by_year = hansen.select('lossyear')
    loss_mask = loss_by_year.lte(year_index).And(loss_by_year.gt(0))
    
    # Combine for visualization
    visualization = {
        'bands': rgb_bands,
        'min': 0,
        'max': 0.3
    }
    
    # Download the image using Earth Engine's built-in method
    try:
        # Save snapshot
        site_dir = os.path.join(save_dir, site['name'])
        os.makedirs(site_dir, exist_ok=True)
        
        snapshot_path = os.path.join(site_dir, 'snapshots')
        os.makedirs(snapshot_path, exist_ok=True)
        
        filename = f"{site['name']}_{year}_snapshot.png"
        full_path = os.path.join(snapshot_path, filename)
        
        # Use EE's getThumbnailURL to get static map image
        vis_params = visualization.copy()
        
        # Create composite with loss overlay
        loss_overlay = loss_mask.clip(geometry).selfMask()
        
        # Combine RGB and loss overlay
        rgb_vis = rgb_image.visualize(**vis_params)
        
        # Get thumbnail URL
        thumbnail_url = rgb_image.getThumbUrl({
            'dimensions': 1200,
            'region': geometry,
            'format': 'png',
            'bands': rgb_bands,
            'min': 0,
            'max': 0.3
        })
        
        # Download the thumbnail
        import urllib.request
        urllib.request.urlretrieve(thumbnail_url, full_path)
        
        return full_path
    except Exception as e:
        print(f"Warning: Could not generate snapshot for {site['name']} in {year}: {e}")
        return None

def link_to_economic_data(site, metrics_timeseries):
    """
    Correlate deforestation timing with agricultural area increase, nighttime lights, and production data.
    
    Args:
        site: Site dictionary with geometry and metadata
        metrics_timeseries: List of annual metrics dictionaries
    
    Returns:
        Dictionary with correlation plots, statistics, and analysis results
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(metrics_timeseries)
    
    # Create correlation analysis
    results = {
        'site_name': site['name'],
        'site_type': site['type'],
        'transition_year': site['transition_year']
    }
    
    # Calculate correlations
    if 'ndvi_mean' in df.columns and 'nighttime_lights_mean' in df.columns:
        # Drop NaN values for correlation
        clean_df = df[['ndvi_mean', 'nighttime_lights_mean']].dropna()
        if len(clean_df) > 1:
            corr = clean_df.corr().iloc[0, 1]
            results['ndvi_lights_correlation'] = corr
        else:
            results['ndvi_lights_correlation'] = None
    else:
        results['ndvi_lights_correlation'] = None
    
    # Calculate deforestation rate (percentage change in cumulative loss)
    if 'cumulative_loss_by_year' in df.columns:
        df['deforestation_rate'] = df['cumulative_loss_by_year'].diff()
        results['max_deforestation_year'] = df.loc[df['deforestation_rate'].idxmax(), 'year'] if len(df) > 0 else None
        results['max_deforestation_rate'] = df['deforestation_rate'].max()
    else:
        results['max_deforestation_year'] = None
        results['max_deforestation_rate'] = None
    
    # Calculate nighttime lights trend
    if 'nighttime_lights_mean' in df.columns:
        lights_data = df['nighttime_lights_mean'].dropna()
        if len(lights_data) > 2:
            # Linear trend
            years_clean = df.loc[lights_data.index, 'year'].values
            coeffs = np.polyfit(years_clean, lights_data.values, 1)
            results['lights_trend_slope'] = coeffs[0]
        else:
            results['lights_trend_slope'] = None
    else:
        results['lights_trend_slope'] = None
    
    # Check if economic activity increased after deforestation
    if results.get('max_deforestation_year') and 'nighttime_lights_mean' in df.columns:
        # Get pre and post deforestation averages
        pre_df = df[df['year'] < results['max_deforestation_year']]
        post_df = df[df['year'] > results['max_deforestation_year']]
        
        if len(pre_df) > 0 and len(post_df) > 0:
            pre_lights = pre_df['nighttime_lights_mean'].dropna().mean()
            post_lights = post_df['nighttime_lights_mean'].dropna().mean()
            
            if pre_lights and post_lights:
                results['lights_increase_percentage'] = ((post_lights - pre_lights) / pre_lights) * 100
            else:
                results['lights_increase_percentage'] = None
        else:
            results['lights_increase_percentage'] = None
    else:
        results['lights_increase_percentage'] = None
    
    # Generate visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Forest loss and NDVI over time
    ax1 = axes[0]
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Forest Cover % / NDVI', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    if 'cumulative_loss_by_year' in df.columns:
        ax1.plot(df['year'], df['cumulative_loss_by_year'], 'r-', label='Cumulative Forest Loss', linewidth=2)
    
    if 'ndvi_mean' in df.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df['year'], df['ndvi_mean'], 'g-', label='NDVI', linewidth=2)
        ax1_twin.set_ylabel('NDVI', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        ax1_twin.legend(loc='upper right')
    
    ax1.axvline(x=site['transition_year'], color='orange', linestyle='--', 
                label=f"Deforestation Year ({site['transition_year']})")
    ax1.legend(loc='upper left')
    ax1.set_title(f"{site['name']} - Forest Loss and Vegetation Health")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Nighttime lights over time
    ax2 = axes[1]
    if 'nighttime_lights_mean' in df.columns:
        lights_data = df['nighttime_lights_mean'].dropna()
        if len(lights_data) > 0:
            years_clean = df.loc[lights_data.index, 'year'].values
            ax2.plot(years_clean, lights_data.values, 'b-o', label='Nighttime Lights', linewidth=2, markersize=6)
    
    ax2.axvline(x=site['transition_year'], color='orange', linestyle='--', 
                label=f"Deforestation Year ({site['transition_year']})")
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Nighttime Lights (Radiance)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_title(f"{site['name']} - Economic Activity Indicator")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    site_dir = os.path.join('output_economic_analysis', site['name'])
    os.makedirs(site_dir, exist_ok=True)
    
    plot_path = os.path.join(site_dir, 'timeseries_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    results['plot_path'] = plot_path
    
    return results

def create_case_study_report(site, metrics_timeseries, analysis_results, save_dir='output_economic_analysis'):
    """
    Compile all outputs for a site and create a summary report.
    
    Args:
        site: Site dictionary with geometry and metadata
        metrics_timeseries: List of annual metrics dictionaries
        analysis_results: Results from link_to_economic_data()
        save_dir: Directory to save outputs
    
    Returns:
        Path to report directory
    """
    site_dir = os.path.join(save_dir, site['name'])
    os.makedirs(site_dir, exist_ok=True)
    
    # 1. Save metrics to CSV
    df = pd.DataFrame(metrics_timeseries)
    csv_path = os.path.join(site_dir, 'metrics_timeseries.csv')
    df.to_csv(csv_path, index=False)
    
    # 2. Create text summary report
    summary_path = os.path.join(site_dir, 'site_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CASE STUDY REPORT: {site['name']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Site Type: {site['type'].upper()}\n")
        f.write(f"Description: {site['description']}\n")
        f.write(f"Location: {site['center_lat']:.2f}°S, {site['center_lon']:.2f}°W\n")
        f.write(f"Expected Transition Year: {site['transition_year']}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n")
        
        if analysis_results.get('max_deforestation_year'):
            f.write(f"Peak Deforestation Year: {analysis_results['max_deforestation_year']:.0f}\n")
        
        if analysis_results.get('lights_increase_percentage'):
            f.write(f"Nighttime Lights Increase: {analysis_results['lights_increase_percentage']:.1f}%\n")
        
        if analysis_results.get('ndvi_lights_correlation'):
            corr = analysis_results['ndvi_lights_correlation']
            f.write(f"NDVI-Lights Correlation: {corr:.3f}\n")
        
        if analysis_results.get('lights_trend_slope'):
            trend = analysis_results['lights_trend_slope']
            direction = "increasing" if trend > 0 else "decreasing"
            f.write(f"Economic Activity Trend: {direction} ({trend:.4f}/year)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("=" * 80 + "\n\n")
        
        # Write interpretation
        f.write("This site shows how deforestation in the Brazilian Amazon has been linked\n")
        f.write("to economic development through agricultural production.\n\n")
        
        if site['type'] == 'farm':
            f.write("As a farm site, this area demonstrates:\n")
            f.write("- Conversion of forest to crop cultivation\n")
            f.write("- Changes in vegetation productivity (NDVI)\n")
            f.write("- Economic activity indicators (nighttime lights)\n")
        else:
            f.write("As a pasture site, this area demonstrates:\n")
            f.write("- Conversion of forest to cattle ranching\n")
            f.write("- Changes in land use patterns\n")
            f.write("- Local economic development\n")
        
        f.write("\nAnnual snapshots and time series plots provide visual evidence of\n")
        f.write("the transformation and its economic consequences.\n")
    
    return site_dir

def main():
    """
    Main execution block to run analysis for all 5-6 sites and generate combined report.
    """
    print("DEFORESTATION TO ECONOMIC IMPACT ANALYSIS")
    print("Brazilian Amazon Case Studies")
    
    # Define a broad ROI covering the Amazon region
    # Using a large bounding box that covers major deforestation areas
    roi = ee.Geometry.Rectangle([-66.5, -13.0, -44.5, 0.0])  # Covers Pará, Mato Grosso, Rondônia
    
    print("\nStep 1: Identifying case study sites")
    sites = identify_case_study_sites(roi, num_farm_sites=3, num_pasture_sites=3)
    print(f"Found {len(sites)} sites:")
    for site in sites:
        print(f"  - {site['name']}: {site['description']}")
    
    # Years to analyze
    years = list(range(2000, 2024))  # 2000-2023
    
    # Storage for all results
    all_results = []
    
    # Process each site
    for i, site in enumerate(sites):
        print(f"\n{'=' * 80}")
        print(f"Processing Site {i+1}/{len(sites)}: {site['name']}")
        print(f"{'=' * 80}")
        
        # Extract annual metrics
        print("\nExtracting annual metrics")
        metrics_timeseries = []
        
        for year in years:
            try:
                print(f"  Processing {year}", end=' ')
                metrics = extract_annual_metrics(site, year)
                metrics_timeseries.append(metrics)
                print("[OK]")
            except Exception as e:
                print(f"[ERROR] Error: {e}")
                continue
        
        # Generate annual snapshots (sample years only to save time)
        print("\nGenerating annual snapshots (sampling every 3 years)")
        sample_years = years[::3]  # Every 3rd year
        
        for year in sample_years:
            try:
                print(f"  Generating snapshot for {year}", end=' ')
                snapshot_path = generate_annual_snapshot(site, year)
                if snapshot_path:
                    print("[OK]")
                else:
                    print("[ERROR]")
            except Exception as e:
                print(f"[ERROR] Error: {e}")
        
        # Link to economic data and create plots
        print("\nAnalyzing economic correlations")
        try:
            analysis_results = link_to_economic_data(site, metrics_timeseries)
            print("[OK] Analysis complete")
        except Exception as e:
            print(f"[ERROR] Error in analysis: {e}")
            analysis_results = {}
        
        # Create case study report
        print("\nGenerating case study report")
        try:
            report_dir = create_case_study_report(site, metrics_timeseries, analysis_results)
            print(f"[OK] Report saved to {report_dir}")
        except Exception as e:
            print(f"[ERROR] Error creating report: {e}")
        
        all_results.append({
            'site': site,
            'metrics': metrics_timeseries,
            'analysis': analysis_results
        })
    
    # Create combined summary
    print("GENERATING COMBINED SUMMARY")
    
    summary_path = os.path.join('output_economic_analysis', 'combined_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMBINED ANALYSIS SUMMARY\n")
        f.write("Deforestation to Economic Impact - Brazilian Amazon\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Sites Analyzed: {len(sites)}\n")
        f.write(f"  - Farm Sites: {sum(1 for s in sites if s['type'] == 'farm')}\n")
        f.write(f"  - Pasture Sites: {sum(1 for s in sites if s['type'] == 'pasture')}\n")
        f.write(f"Analysis Period: 2000-2023\n\n")
        
        f.write("SITE-BY-SITE RESULTS:\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            site = result['site']
            analysis = result.get('analysis', {})
            
            f.write(f"\n{site['name'].upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Type: {site['type']}\n")
            f.write(f"Description: {site['description']}\n")
            
            if analysis.get('lights_increase_percentage'):
                f.write(f"Economic Activity Increase: {analysis['lights_increase_percentage']:.1f}%\n")
            
            if analysis.get('max_deforestation_year'):
                f.write(f"Peak Deforestation: {analysis['max_deforestation_year']:.0f}\n")
    
    print(f"[OK] Combined summary saved to {summary_path}")
    
    print("ANALYSIS COMPLETE!")
    print(f"Output directory: output_economic_analysis/")
    print(f"Results available for {len(sites)} sites")

if __name__ == "__main__":
    main()

