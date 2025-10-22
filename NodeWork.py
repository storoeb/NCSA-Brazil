import ee
import os
import geemap
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from datetime import datetime
from sklearn.cluster import DBSCAN

try:
    ee.Initialize()
except:
    ee.Initialize(project="fluent-cosine-473703-g7")

def create_grid_nodes(roi, grid_size_km=10):
    """
    Divide ROI into grid cells, each becomes a node
    
    Args:
        roi: Earth Engine Geometry
        grid_size_km: Size of each grid cell in km
    
    Returns:
        list of node dictionaries with geometry and id
    """
    bounds = roi.bounds().coordinates().getInfo()[0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]
    
    # Convert km to degrees (approximate) 
    grid_size_deg = grid_size_km / 111.0
    
    nodes = []
    node_id = 0
    
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            # Create cell geometry
            cell = ee.Geometry.Rectangle([lon, lat, 
                                         lon + grid_size_deg, 
                                         lat + grid_size_deg])
            
            nodes.append({
                'id': node_id,
                'geometry': cell,
                'lon': lon + grid_size_deg/2,
                'lat': lat + grid_size_deg/2,
                'row': int((lat - min_lat) / grid_size_deg),
                'col': int((lon - min_lon) / grid_size_deg)
            })
            
            node_id += 1
            lon += grid_size_deg
        lat += grid_size_deg
    
    return nodes

def extract_soil_metrics(geometry):
    """
    Extract soil properties from SoilGrids for a given geometry
    
    Args:
        geometry: Earth Engine Geometry
    
    Returns:
        dict with soil properties: soc, clay, silt, sand, ph, cec
    """
    try:
        # Load SoilGrids datasets (topsoil: 0-5cm depth)
        # Band names follow format: property_0-5cm_mean
        
        # Soil Organic Carbon (dg/kg, so divide by 10 to get g/kg)
        soc = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean')
        
        # Clay content (g/kg, divide by 10 to get %)
        clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
        
        # Silt content (g/kg, divide by 10 to get %)
        silt = ee.Image("projects/soilgrids-isric/silt_mean").select('silt_0-5cm_mean')
        
        # Sand content (g/kg, divide by 10 to get %)
        sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')
        
        # pH in H2O (pH * 10, so divide by 10)
        phh2o = ee.Image("projects/soilgrids-isric/phh2o_mean").select('phh2o_0-5cm_mean')
        
        # Cation Exchange Capacity (mmol(c)/kg, divide by 10)
        cec = ee.Image("projects/soilgrids-isric/cec_mean").select('cec_0-5cm_mean')
        
        # Extract all soil properties in one call
        soil_data = ee.Image([soc, clay, silt, sand, phh2o, cec]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=250,  # 250m resolution
            maxPixels=1e9
        ).getInfo()
        
        return {
            'soil_organic_carbon': soil_data.get('soc_0-5cm_mean', 0) / 10.0 if soil_data.get('soc_0-5cm_mean') else 0,
            'clay_content': soil_data.get('clay_0-5cm_mean', 0) / 10.0 if soil_data.get('clay_0-5cm_mean') else 0,
            'silt_content': soil_data.get('silt_0-5cm_mean', 0) / 10.0 if soil_data.get('silt_0-5cm_mean') else 0,
            'sand_content': soil_data.get('sand_0-5cm_mean', 0) / 10.0 if soil_data.get('sand_0-5cm_mean') else 0,
            'soil_ph': soil_data.get('phh2o_0-5cm_mean', 0) / 10.0 if soil_data.get('phh2o_0-5cm_mean') else 0,
            'cation_exchange_capacity': soil_data.get('cec_0-5cm_mean', 0) / 10.0 if soil_data.get('cec_0-5cm_mean') else 0
        }
    except Exception as e:
        print(f"  Warning: Could not extract soil metrics - {e}")
        return {
            'soil_organic_carbon': 0,
            'clay_content': 0,
            'silt_content': 0,
            'sand_content': 0,
            'soil_ph': 0,
            'cation_exchange_capacity': 0
        }

def extract_node_metrics(node, start_date='2000-01-01', end_date='2024-01-01'):
    """
    Extract forest metrics for a node
    """
    geometry = node['geometry']
    
    # 1. Forest Loss Detection (Hansen)
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    
    # Get forest cover in 2000
    treecover = hansen.select('treecover2000').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30
    ).getInfo()
    
    # Get loss year
    lossyear = hansen.select('lossyear').reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=geometry,
        scale=30
    ).getInfo()
    
    # 2. NDVI Time Series
    ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .select('NDVI') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)

    # Calculate NDVI time series
    def add_ndvi_stats(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=250
        )
        return ee.Feature(None, {'NDVI': stats.get('NDVI')})

    ndvi_features = ndvi_collection.map(add_ndvi_stats)
    ndvi_series = ndvi_features.aggregate_array('NDVI').getInfo()
    
    # Calculate NDVI metrics
    ndvi_values = [v * 0.0001 for v in ndvi_series if v is not None]
    ndvi_mean = np.mean(ndvi_values) if ndvi_values else 0
    ndvi_trend = np.polyfit(range(len(ndvi_values)), ndvi_values, 1)[0] if len(ndvi_values) > 1 else 0
    
    loss_value = lossyear.get('lossyear', 0)
    loss_year = 2000 + loss_value if loss_value else None
    
    # 3. Extract soil metrics
    soil_metrics = extract_soil_metrics(geometry)
    
    # Merge all metrics
    metrics = {
        'node_id': node['id'],
        'forest_cover_2000': treecover.get('treecover2000', 0),
        'loss_year': loss_year,
        'ndvi_mean': ndvi_mean,
        'ndvi_trend': ndvi_trend,
        'has_loss': loss_year is not None
    }
    metrics.update(soil_metrics)
    
    return metrics

def build_forest_graph(nodes, metrics_list):
    """
    Build NetworkX graph where nodes are forest patches and edges are neighbors
    
    Answers your question: "when do neighbor nodes spawn?"
    - They spawn based on grid position (row, col)
    """
    G = nx.Graph()
    
    # Add nodes with attributes
    for node, metrics in zip(nodes, metrics_list):
        G.add_node(
            node['id'],
            pos=(node['lon'], node['lat']),
            row=node['row'],
            col=node['col'],
            **metrics
        )
    
    # Add edges between neighbors (4-connectivity or 8-connectivity)
    for node in nodes:
        node_id = node['id']
        row, col = node['row'], node['col']
        
        # Check 4 neighbors (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row, neighbor_col = row + dr, col + dc
            
            # Find neighbor node
            for other_node in nodes:
                if other_node['row'] == neighbor_row and other_node['col'] == neighbor_col:
                    G.add_edge(node_id, other_node['id'])
                    break
    
    return G

def analyze_deforestation_spread(G):
    """
    Analyze how deforestation spreads through the graph
    """
    # Get nodes with forest loss
    loss_nodes = [(n, data['loss_year']) for n, data in G.nodes(data=True) 
                  if data.get('has_loss', False)]
    
    if not loss_nodes:
        return None
    
    # Sort by loss year
    loss_nodes.sort(key=lambda x: x[1])
    
    # Track spread patterns
    spread_analysis = {
        'total_loss_nodes': len(loss_nodes),
        'earliest_loss': min(n[1] for n in loss_nodes),
        'latest_loss': max(n[1] for n in loss_nodes),
        'neighbor_cascade': []
    }
    
    # Check if loss spreads to neighbors
    for node_id, year in loss_nodes:
        neighbors = list(G.neighbors(node_id))
        neighbor_losses = [(n, G.nodes[n]['loss_year']) 
                          for n in neighbors 
                          if G.nodes[n].get('has_loss', False)]
        
        # Count neighbors that lost forest AFTER this node
        subsequent_losses = [n for n, y in neighbor_losses if y and y > year]
        
        if subsequent_losses:
            spread_analysis['neighbor_cascade'].append({
                'node': node_id,
                'year': year,
                'subsequent_neighbor_losses': len(subsequent_losses)
            })
    
    return spread_analysis

def analyze_soil_deforestation_correlation(G):
    """
    Analyze correlation between soil properties and deforestation
    
    Args:
        G: NetworkX graph with node attributes
    
    Returns:
        dict with correlation analysis results
    """
    # Separate nodes into loss vs intact
    loss_nodes = [n for n, data in G.nodes(data=True) if data.get('has_loss', False)]
    intact_nodes = [n for n, data in G.nodes(data=True) if not data.get('has_loss', False)]
    
    if not loss_nodes or not intact_nodes:
        return None
    
    # Soil properties to analyze
    soil_properties = ['soil_organic_carbon', 'clay_content', 'silt_content', 
                      'sand_content', 'soil_ph', 'cation_exchange_capacity']
    
    analysis = {
        'loss_nodes_count': len(loss_nodes),
        'intact_nodes_count': len(intact_nodes),
        'soil_comparisons': {}
    }
    
    # Compare soil properties between loss and intact nodes
    for prop in soil_properties:
        loss_values = [G.nodes[n].get(prop, 0) for n in loss_nodes]
        intact_values = [G.nodes[n].get(prop, 0) for n in intact_nodes]
        
        loss_mean = np.mean(loss_values) if loss_values else 0
        intact_mean = np.mean(intact_values) if intact_values else 0
        
        analysis['soil_comparisons'][prop] = {
            'loss_mean': loss_mean,
            'intact_mean': intact_mean,
            'difference': loss_mean - intact_mean,
            'percent_difference': ((loss_mean - intact_mean) / intact_mean * 100) if intact_mean != 0 else 0
        }
    
    # Calculate correlations between soil properties and forest metrics
    all_nodes = list(G.nodes())
    forest_cover = [G.nodes[n].get('forest_cover_2000', 0) for n in all_nodes]
    
    analysis['correlations'] = {}
    for prop in soil_properties:
        soil_values = [G.nodes[n].get(prop, 0) for n in all_nodes]
        if len(soil_values) > 1 and np.std(soil_values) > 0:
            correlation = np.corrcoef(soil_values, forest_cover)[0, 1]
            analysis['correlations'][prop] = correlation
        else:
            analysis['correlations'][prop] = 0
    
    return analysis

def visualize_forest_graph(G, highlight_loss=True, save_path=None):
    """
    Visualize the forest graph with soil properties and save to file
    
    Args:
        G: NetworkX graph
        highlight_loss: Boolean to color nodes by loss status
        save_path: Path to save the visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Panel 1: Forest Loss Status
    node_colors_loss = ['red' if G.nodes[n].get('has_loss', False) else 'green' 
                        for n in G.nodes()]
    nx.draw(G, pos, node_color=node_colors_loss, node_size=150, 
            ax=ax1, with_labels=False, edge_color='gray', alpha=0.6)
    ax1.set_title('Forest Loss Status (Red=Loss, Green=Intact)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Panel 2: Loss Timeline
    loss_data = [(G.nodes[n]['loss_year'], n) 
                 for n in G.nodes() if G.nodes[n].get('has_loss', False)]
    
    if loss_data:
        years, nodes = zip(*sorted(loss_data))
        counts = {}
        for year in years:
            counts[year] = counts.get(year, 0) + 1
        
        ax2.bar(counts.keys(), counts.values(), color='red', alpha=0.7)
        ax2.set_title('Forest Loss Timeline', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Nodes with Loss')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Forest Loss Data', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Forest Loss Timeline', fontsize=12, fontweight='bold')
    
    # Panel 3: Soil Organic Carbon
    soc_values = [G.nodes[n].get('soil_organic_carbon', 0) for n in G.nodes()]
    if max(soc_values) > 0:
        nodes_soc = nx.draw_networkx_nodes(G, pos, node_color=soc_values, 
                                           node_size=150, ax=ax3, cmap='YlOrBr',
                                           vmin=min(soc_values), vmax=max(soc_values))
        nx.draw_networkx_edges(G, pos, ax=ax3, edge_color='gray', alpha=0.3)
        plt.colorbar(nodes_soc, ax=ax3, label='SOC (g/kg)')
        ax3.set_title('Soil Organic Carbon Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
    else:
        ax3.text(0.5, 0.5, 'No Soil Organic Carbon Data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Soil Organic Carbon Distribution', fontsize=12, fontweight='bold')
    
    # Panel 4: Clay Content
    clay_values = [G.nodes[n].get('clay_content', 0) for n in G.nodes()]
    if max(clay_values) > 0:
        nodes_clay = nx.draw_networkx_nodes(G, pos, node_color=clay_values,
                                            node_size=150, ax=ax4, cmap='RdYlBu_r',
                                            vmin=min(clay_values), vmax=max(clay_values))
        nx.draw_networkx_edges(G, pos, ax=ax4, edge_color='gray', alpha=0.3)
        plt.colorbar(nodes_clay, ax=ax4, label='Clay Content (%)')
        ax4.set_title('Clay Content Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
    else:
        ax4.text(0.5, 0.5, 'No Clay Content Data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Clay Content Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # ROI
    coordinates = [
        (-62.527222, -9.235),
        (-63.279167, -10.252222),
        (-54.679444, -2.519722),
        (-60.182500, -10.244722),
    ]
    
    # Create ROI from convex hull of points
    roi = ee.Geometry.Polygon([coordinates])
    
    print("Step 1: Creating grid nodes")
    nodes = create_grid_nodes(roi, grid_size_km=10)
    print(f"Created {len(nodes)} nodes")
    
    # Set number of nodes to process (recommended: 50-100 for reasonable processing time)
    num_nodes = 100  # Change this value: 64 for 8x8, 100 for 10x10, or len(nodes) for all
    
    print(f"\nStep 2: Extracting metrics for each node (this will take ~{num_nodes * 6 // 60} minutes)")
    metrics_list = []
    for i, node in enumerate(nodes[:num_nodes]):
        print(f"Processing node {i+1}/{num_nodes}")
        metrics = extract_node_metrics(node)
        metrics_list.append(metrics)
        print(f"  - Forest cover: {metrics['forest_cover_2000']:.1f}%")
        print(f"  - Loss year: {metrics['loss_year']}")
        print(f"  - Soil Organic Carbon: {metrics['soil_organic_carbon']:.1f} g/kg")
        print(f"  - Clay content: {metrics['clay_content']:.1f}%")
    
    print("\nStep 3: Building graph")
    G = build_forest_graph(nodes[:num_nodes], metrics_list)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nStep 4: Analyzing deforestation spread")
    spread = analyze_deforestation_spread(G)
    if spread:
        print(f"Total nodes with loss: {spread['total_loss_nodes']}")
        print(f"Loss period: {spread['earliest_loss']} - {spread['latest_loss']}")
        print(f"Nodes with neighbor cascade: {len(spread['neighbor_cascade'])}")
    
    print("\nStep 4.5: Analyzing soil-deforestation correlations")
    soil_analysis = analyze_soil_deforestation_correlation(G)
    if soil_analysis:
        print(f"Nodes with forest loss: {soil_analysis['loss_nodes_count']}")
        print(f"Intact forest nodes: {soil_analysis['intact_nodes_count']}")
        print("\nSoil Property Comparisons (Loss vs Intact):")
        for prop, stats in soil_analysis['soil_comparisons'].items():
            print(f"  {prop}:")
            print(f"    Loss nodes: {stats['loss_mean']:.2f}")
            print(f"    Intact nodes: {stats['intact_mean']:.2f}")
            print(f"    Difference: {stats['difference']:.2f} ({stats['percent_difference']:.1f}%)")
        print("\nCorrelations with Forest Cover:")
        for prop, corr in soil_analysis['correlations'].items():
            print(f"  {prop}: {corr:.3f}")
    
    print("\nStep 5: Visualizing and saving plots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"output_graphs/forest_soil_analysis_{timestamp}.png"
    visualize_forest_graph(G, save_path=save_path)
    print(f"Visualization saved to {save_path}")