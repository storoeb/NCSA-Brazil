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
    ee.Initialize(project="dotted-saga-475001-n3")

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
    
    return {
        'node_id': node['id'],
        'forest_cover_2000': treecover.get('treecover2000', 0),
        'loss_year': loss_year,
        'ndvi_mean': ndvi_mean,
        'ndvi_trend': ndvi_trend,
        'has_loss': loss_year is not None
    }

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

def visualize_forest_graph(G, highlight_loss=True, save_path=None):
    """
    Visualize the forest graph and save to file
    
    Args:
        G: NetworkX graph
        highlight_loss: Boolean to color nodes by loss status
        save_path: Path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Color nodes by loss status
    if highlight_loss:
        node_colors = ['red' if G.nodes[n].get('has_loss', False) else 'green' 
                      for n in G.nodes()]
        title1 = 'Forest Loss Status (Red=Loss, Green=Intact)'
    else:
        ndvi_values = [G.nodes[n].get('ndvi_mean', 0) for n in G.nodes()]
        node_colors = ndvi_values
        title1 = 'NDVI Values'
    
    # Draw graph
    nx.draw(G, pos, node_color=node_colors, node_size=100, 
            ax=ax1, with_labels=False, edge_color='gray', alpha=0.6)
    ax1.set_title(title1)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plot loss timeline
    loss_data = [(G.nodes[n]['loss_year'], n) 
                 for n in G.nodes() if G.nodes[n].get('has_loss', False)]
    
    if loss_data:
        years, nodes = zip(*sorted(loss_data))
        counts = {}
        for year in years:
            counts[year] = counts.get(year, 0) + 1
        
        ax2.bar(counts.keys(), counts.values(), color='red', alpha=0.7)
        ax2.set_title('Forest Loss Timeline')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Nodes with Loss')
        ax2.grid(True, alpha=0.3)
    
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
    
    print("\nStep 2: Extracting metrics for each node (this may take a while)")
    metrics_list = []
    for i, node in enumerate(nodes[:10]):  # Limit to first 10 for testing
        print(f"Processing node {i+1}/{min(10, len(nodes))}")
        metrics = extract_node_metrics(node)
        metrics_list.append(metrics)
        print(f"  - Forest cover: {metrics['forest_cover_2000']:.1f}%")
        print(f"  - Loss year: {metrics['loss_year']}")
    
    print("\nStep 3: Building graph")
    G = build_forest_graph(nodes[:10], metrics_list)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nStep 4: Analyzing deforestation spread")
    spread = analyze_deforestation_spread(G)
    if spread:
        print(f"Total nodes with loss: {spread['total_loss_nodes']}")
        print(f"Loss period: {spread['earliest_loss']} - {spread['latest_loss']}")
        print(f"Nodes with neighbor cascade: {len(spread['neighbor_cascade'])}")
    
    print("\nStep 5: Visualizing and saving plots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"output_graphs/forest_analysis_{timestamp}.png"
    visualize_forest_graph(G, save_path=save_path)
    print(f"Visualization saved to {save_path}")