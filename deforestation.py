import ee
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from collections import deque
import json
import time

try:
    ee.Initialize()
except:
    ee.Initialize(project="fluent-cosine-473703-g7")


class Node:
    """Represents a 250m x 250m chunk of land"""
    
    def __init__(self, lon, lat, node_id):
        self.lon = lon
        self.lat = lat
        self.node_id = node_id
        self.geometry = ee.Geometry.Point([lon, lat]).buffer(125)  # 250m x 250m square
        
        # Data attributes
        self.deforestation_year = None
        self.tree_cover_2000 = None
        self.ndvi_data = []
        self.ndvi_dates = []
        self.neighbors = []
        
        # Soil attributes
        self.soil_organic_carbon = None
        self.clay_content = None
        self.silt_content = None
        self.sand_content = None
        self.soil_ph = None
        self.cation_exchange_capacity = None
        
        # Status tracking
        self.is_deforested = False
        self.status_timeline = []  # Track status changes over time
        
    def fetch_hansen_data(self):
        """Fetch Hansen deforestation data for this node"""
        try:
            hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
            
            # Get loss year and tree cover
            result = hansen.select(['lossyear', 'treecover2000']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.geometry,
                scale=30,
                bestEffort=True
            ).getInfo()
            
            loss_value = result.get('lossyear', 0)
            self.tree_cover_2000 = result.get('treecover2000', 0)
            
            if loss_value and loss_value > 0:
                self.deforestation_year = 2000 + int(loss_value)
                self.is_deforested = True
            
            return True
        except Exception as e:
            print(f"Error fetching Hansen data for node {self.node_id}: {e}")
            return False
    
    def fetch_ndvi_timeseries(self, start_date='2000-01-01', end_date='2024-12-31'):
        """Fetch NDVI time series for this node"""
        try:
            ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
                .select('NDVI') \
                .filterDate(start_date, end_date) \
                .filterBounds(self.geometry)
            
            def get_ndvi(image):
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
                mean_ndvi = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.geometry,
                    scale=250
                ).get('NDVI')
                return ee.Feature(None, {'date': date, 'NDVI': mean_ndvi})
            
            ndvi_features = ndvi_collection.map(get_ndvi).filter(
                ee.Filter.notNull(['NDVI'])
            )
            
            self.ndvi_dates = ndvi_features.aggregate_array('date').getInfo()
            ndvi_values = ndvi_features.aggregate_array('NDVI').getInfo()
            
            # Scale NDVI values (MODIS NDVI is scaled by 10000)
            self.ndvi_data = [v * 0.0001 for v in ndvi_values]
            
            return True
        except Exception as e:
            print(f"Error fetching NDVI data for node {self.node_id}: {e}")
            return False
    
    def fetch_soil_data(self):
        """Fetch soil properties from SoilGrids for this node"""
        try:
            # Load SoilGrids datasets (topsoil: 0-5cm depth)
            soc = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean')
            clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
            silt = ee.Image("projects/soilgrids-isric/silt_mean").select('silt_0-5cm_mean')
            sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')
            phh2o = ee.Image("projects/soilgrids-isric/phh2o_mean").select('phh2o_0-5cm_mean')
            cec = ee.Image("projects/soilgrids-isric/cec_mean").select('cec_0-5cm_mean')
            
            # Extract all soil properties in one call
            soil_data = ee.Image([soc, clay, silt, sand, phh2o, cec]).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.geometry,
                scale=250,
                bestEffort=True
            ).getInfo()
            
            # Store soil properties (all scaled by 10, so divide by 10)
            self.soil_organic_carbon = soil_data.get('soc_0-5cm_mean', 0) / 10.0 if soil_data.get('soc_0-5cm_mean') else 0
            self.clay_content = soil_data.get('clay_0-5cm_mean', 0) / 10.0 if soil_data.get('clay_0-5cm_mean') else 0
            self.silt_content = soil_data.get('silt_0-5cm_mean', 0) / 10.0 if soil_data.get('silt_0-5cm_mean') else 0
            self.sand_content = soil_data.get('sand_0-5cm_mean', 0) / 10.0 if soil_data.get('sand_0-5cm_mean') else 0
            self.soil_ph = soil_data.get('phh2o_0-5cm_mean', 0) / 10.0 if soil_data.get('phh2o_0-5cm_mean') else 0
            self.cation_exchange_capacity = soil_data.get('cec_0-5cm_mean', 0) / 10.0 if soil_data.get('cec_0-5cm_mean') else 0
            
            return True
        except Exception as e:
            print(f"Error fetching soil data for node {self.node_id}: {e}")
            return False
    
    def classify_ndvi_status(self, ndvi_value):
        """Classify land status based on NDVI value"""
        if ndvi_value > 0.6:
            return "Forest/Dense Vegetation"
        elif ndvi_value > 0.3:
            return "Agriculture/Regrowth"
        else:
            return "Bare Ground/Dirt"
    
    def get_ndvi_at_year(self, year):
        """Get average NDVI for a specific year"""
        year_ndvi = []
        for date_str, ndvi in zip(self.ndvi_dates, self.ndvi_data):
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if date.year == year:
                year_ndvi.append(ndvi)
        
        return np.mean(year_ndvi) if year_ndvi else None
    
    def get_post_deforestation_trajectory(self):
        """Analyze NDVI trajectory after deforestation"""
        if not self.is_deforested or not self.deforestation_year:
            return None
        
        trajectory = []
        for date_str, ndvi in zip(self.ndvi_dates, self.ndvi_data):
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if date.year >= self.deforestation_year:
                trajectory.append({
                    'date': date,
                    'ndvi': ndvi,
                    'status': self.classify_ndvi_status(ndvi),
                    'years_since_deforestation': date.year - self.deforestation_year
                })
        
        return trajectory
    
    def compute_ndvi_statistics(self):
        """Compute NDVI statistics for this node"""
        if not self.ndvi_data:
            return None
        
        return {
            'mean': np.mean(self.ndvi_data),
            'std': np.std(self.ndvi_data),
            'min': np.min(self.ndvi_data),
            'max': np.max(self.ndvi_data),
            'trend': self._compute_ndvi_trend()
        }
    
    def _compute_ndvi_trend(self):
        """Compute linear trend of NDVI over time"""
        if len(self.ndvi_data) < 2:
            return 0
        
        x = np.arange(len(self.ndvi_data))
        y = np.array(self.ndvi_data)
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # Slope
    
    def __repr__(self):
        return f"Node({self.node_id}, lon={self.lon:.4f}, lat={self.lat:.4f}, deforested={self.is_deforested}, year={self.deforestation_year})"


class DeforestationTracker:
    """Main tracker class for analyzing deforestation spread"""
    
    # Approximate degrees for 250m spacing (varies by latitude)
    SPACING_DEG = 0.00225
    
    def __init__(self, seed_lon, seed_lat, max_radius_km=5, name="Region"):
        self.seed_lon = seed_lon
        self.seed_lat = seed_lat
        self.max_radius_km = max_radius_km
        self.name = name
        
        self.nodes = {}  # Dictionary: (lon, lat) -> Node
        self.node_counter = 0
        self.deforested_nodes = []
        
    def _get_neighbor_coords(self, lon, lat):
        """Get coordinates of 8 neighboring chunks"""
        spacing = self.SPACING_DEG
        # Adjust spacing for latitude (longitude spacing decreases near poles)
        lon_spacing = spacing / np.cos(np.radians(lat))
        
        neighbors = [
            (lon, lat + spacing),              # N
            (lon, lat - spacing),              # S
            (lon + lon_spacing, lat),          # E
            (lon - lon_spacing, lat),          # W
            (lon + lon_spacing, lat + spacing), # NE
            (lon - lon_spacing, lat + spacing), # NW
            (lon + lon_spacing, lat - spacing), # SE
            (lon - lon_spacing, lat - spacing)  # SW
        ]
        
        return neighbors
    
    def _distance_km(self, lon1, lat1, lon2, lat2):
        """Calculate approximate distance in km between two points"""
        # Simple approximation
        lat_diff = (lat2 - lat1) * 111.0  # 1 degree lat ≈ 111 km
        lon_diff = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def _round_coords(self, lon, lat):
        """Round coordinates to avoid floating point issues"""
        return (round(lon, 6), round(lat, 6))
    
    def explore_region(self, fetch_ndvi=True, max_nodes=500, timeout_minutes=15):
        """
        Explore region starting from seed coordinate using BFS.
        Only includes nodes that are deforested.
        
        Args:
            fetch_ndvi: Whether to fetch NDVI time series for each node
            max_nodes: Maximum number of deforested nodes to find before stopping
            timeout_minutes: Maximum time in minutes before stopping exploration
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Exploring region: {self.name}")
        print(f"Seed: ({self.seed_lon:.4f}, {self.seed_lat:.4f})")
        print(f"Max radius: {self.max_radius_km} km")
        print(f"Max nodes: {max_nodes}")
        print(f"Timeout: {timeout_minutes} minutes")
        print(f"{'='*60}\n")
        
        # BFS queue: (lon, lat, parent_node_id)
        queue = deque([(self.seed_lon, self.seed_lat, None)])
        visited = set()
        
        while queue:
            # Check timeout
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes > timeout_minutes:
                print(f"\n⚠️  TIMEOUT REACHED ({timeout_minutes} minutes)")
                print(f"   Stopping exploration with {len(self.deforested_nodes)} nodes found.\n")
                break
            
            # Check node count limit
            if len(self.deforested_nodes) >= max_nodes:
                print(f"\n⚠️  MAX NODE COUNT REACHED ({max_nodes} nodes)")
                print(f"   Stopping exploration.\n")
                break
            lon, lat, parent_id = queue.popleft()
            
            # Round coordinates to avoid duplicates
            coords = self._round_coords(lon, lat)
            
            # Skip if already visited
            if coords in visited:
                continue
            
            # Skip if too far from seed
            distance = self._distance_km(self.seed_lon, self.seed_lat, lon, lat)
            if distance > self.max_radius_km:
                continue
            
            visited.add(coords)
            
            # Create node
            node = Node(coords[0], coords[1], self.node_counter)
            self.node_counter += 1
            
            # Progress indicator every 10 checked locations
            if self.node_counter > 0 and self.node_counter % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"[Progress: {self.node_counter} locations checked, {len(self.deforested_nodes)} deforested nodes found, {elapsed:.1f} min elapsed]")
            
            # Fetch Hansen data
            print(f"Processing node {node.node_id}: ({coords[0]:.4f}, {coords[1]:.4f})", end=" ")
            success = node.fetch_hansen_data()
            
            if not success:
                print("❌ Failed")
                continue
            
            # Only add nodes that are deforested
            if node.is_deforested:
                print(f"✓ Deforested in {node.deforestation_year}")
                
                # Fetch NDVI if requested
                if fetch_ndvi:
                    node.fetch_ndvi_timeseries()
                
                # Fetch soil data
                node.fetch_soil_data()
                
                self.nodes[coords] = node
                self.deforested_nodes.append(node)
                
                # Add neighbors to queue
                for neighbor_coords in self._get_neighbor_coords(coords[0], coords[1]):
                    neighbor_rounded = self._round_coords(*neighbor_coords)
                    if neighbor_rounded not in visited:
                        queue.append((*neighbor_coords, node.node_id))
            else:
                print("○ Not deforested (skipped)")
        
        elapsed_total = (time.time() - start_time) / 60
        print(f"\n✓ Exploration complete: Found {len(self.nodes)} deforested nodes")
        print(f"  Total locations checked: {self.node_counter}")
        print(f"  Time elapsed: {elapsed_total:.2f} minutes\n")
        
        # Build neighbor relationships
        self._build_neighbor_relationships()
    
    def _build_neighbor_relationships(self):
        """Build neighbor relationships between nodes"""
        for coords, node in self.nodes.items():
            neighbor_coords = self._get_neighbor_coords(coords[0], coords[1])
            for nc in neighbor_coords:
                nc_rounded = self._round_coords(*nc)
                if nc_rounded in self.nodes:
                    node.neighbors.append(self.nodes[nc_rounded])
    
    def analyze_temporal_spread(self):
        """Analyze when nodes were deforested over time"""
        if not self.deforested_nodes:
            return None
        
        years = [n.deforestation_year for n in self.deforested_nodes if n.deforestation_year]
        
        if not years:
            return None
        
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            'first_year': min(years),
            'last_year': max(years),
            'duration': max(years) - min(years),
            'year_counts': year_counts,
            'total_nodes': len(self.deforested_nodes)
        }
    
    def analyze_spatial_pattern(self):
        """Analyze spatial distribution and spread direction"""
        if not self.deforested_nodes:
            return None
        
        lons = [n.lon for n in self.deforested_nodes]
        lats = [n.lat for n in self.deforested_nodes]
        
        return {
            'center_lon': np.mean(lons),
            'center_lat': np.mean(lats),
            'lon_range': (min(lons), max(lons)),
            'lat_range': (min(lats), max(lats)),
            'spread_lon': max(lons) - min(lons),
            'spread_lat': max(lats) - min(lats),
        }
    
    def analyze_ndvi_patterns(self):
        """Analyze NDVI patterns across all nodes"""
        if not self.deforested_nodes:
            return None
        
        results = {
            'nodes_with_data': 0,
            'mean_ndvi_overall': [],
            'post_deforestation_recovery': [],
            'stable_agriculture': 0,
            'abandoned_land': 0,
            'recovering_land': 0
        }
        
        for node in self.deforested_nodes:
            if not node.ndvi_data:
                continue
            
            results['nodes_with_data'] += 1
            results['mean_ndvi_overall'].extend(node.ndvi_data)
            
            # Analyze post-deforestation trajectory
            trajectory = node.get_post_deforestation_trajectory()
            if trajectory:
                # Look at NDVI 3+ years after deforestation
                late_trajectory = [t for t in trajectory if t['years_since_deforestation'] >= 3]
                
                if late_trajectory:
                    avg_late_ndvi = np.mean([t['ndvi'] for t in late_trajectory])
                    
                    if avg_late_ndvi > 0.4:  # Maintaining vegetation
                        results['stable_agriculture'] += 1
                    elif avg_late_ndvi < 0.25:  # Staying bare
                        results['abandoned_land'] += 1
                    else:
                        results['recovering_land'] += 1
        
        if results['mean_ndvi_overall']:
            results['mean_ndvi'] = np.mean(results['mean_ndvi_overall'])
            results['std_ndvi'] = np.std(results['mean_ndvi_overall'])
        
        return results
    
    def analyze_neighbor_similarity(self):
        """Analyze NDVI similarity between neighboring nodes"""
        similarities = []
        
        for node in self.deforested_nodes:
            if not node.ndvi_data or not node.neighbors:
                continue
            
            for neighbor in node.neighbors:
                if not neighbor.ndvi_data:
                    continue
                
                # Compare NDVI trajectories
                correlation = self._compute_ndvi_correlation(node, neighbor)
                if correlation is not None:
                    similarities.append({
                        'node1': node.node_id,
                        'node2': neighbor.node_id,
                        'correlation': correlation,
                        'year_diff': abs((node.deforestation_year or 0) - (neighbor.deforestation_year or 0))
                    })
        
        return similarities
    
    def _compute_ndvi_correlation(self, node1, node2):
        """Compute correlation between NDVI time series of two nodes"""
        # Align dates
        dates1 = set(node1.ndvi_dates)
        dates2 = set(node2.ndvi_dates)
        common_dates = sorted(dates1.intersection(dates2))
        
        if len(common_dates) < 10:  # Need sufficient data
            return None
        
        ndvi1 = [node1.ndvi_data[node1.ndvi_dates.index(d)] for d in common_dates]
        ndvi2 = [node2.ndvi_data[node2.ndvi_dates.index(d)] for d in common_dates]
        
        return np.corrcoef(ndvi1, ndvi2)[0, 1]
    
    def analyze_soil_patterns(self):
        """Analyze soil properties across deforested nodes"""
        if not self.deforested_nodes:
            return None
        
        # Collect soil data
        soil_props = {
            'soil_organic_carbon': [],
            'clay_content': [],
            'silt_content': [],
            'sand_content': [],
            'soil_ph': [],
            'cation_exchange_capacity': []
        }
        
        years = []
        tree_cover = []
        
        for node in self.deforested_nodes:
            if node.soil_organic_carbon is not None and node.soil_organic_carbon > 0:
                soil_props['soil_organic_carbon'].append(node.soil_organic_carbon)
                soil_props['clay_content'].append(node.clay_content)
                soil_props['silt_content'].append(node.silt_content)
                soil_props['sand_content'].append(node.sand_content)
                soil_props['soil_ph'].append(node.soil_ph)
                soil_props['cation_exchange_capacity'].append(node.cation_exchange_capacity)
                years.append(node.deforestation_year)
                tree_cover.append(node.tree_cover_2000)
        
        if not soil_props['soil_organic_carbon']:
            return None
        
        # Calculate statistics and correlations
        analysis = {
            'statistics': {},
            'correlations': {}
        }
        
        for prop, values in soil_props.items():
            if values:
                analysis['statistics'][prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
                # Correlate with tree cover
                if len(values) > 1 and np.std(values) > 0:
                    corr = np.corrcoef(values, tree_cover)[0, 1]
                    analysis['correlations'][f'{prop}_vs_tree_cover'] = corr
        
        return analysis
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*60}")
        print(f"DEFORESTATION ANALYSIS REPORT: {self.name}")
        print(f"{'='*60}\n")
        
        # Temporal spread
        temporal = self.analyze_temporal_spread()
        if temporal:
            print("TEMPORAL SPREAD:")
            print(f"  First deforestation: {temporal['first_year']}")
            print(f"  Last deforestation: {temporal['last_year']}")
            print(f"  Duration: {temporal['duration']} years")
            print(f"  Total nodes: {temporal['total_nodes']}")
            print(f"  Deforestation by year:")
            for year in sorted(temporal['year_counts'].keys()):
                print(f"    {year}: {temporal['year_counts'][year]} nodes")
            print()
        
        # Spatial pattern
        spatial = self.analyze_spatial_pattern()
        if spatial:
            print("SPATIAL PATTERN:")
            print(f"  Center: ({spatial['center_lon']:.4f}, {spatial['center_lat']:.4f})")
            print(f"  Longitude spread: {spatial['spread_lon']:.4f}° ({spatial['spread_lon']*111:.2f} km)")
            print(f"  Latitude spread: {spatial['spread_lat']:.4f}° ({spatial['spread_lat']*111:.2f} km)")
            print()
        
        # NDVI patterns
        ndvi_analysis = self.analyze_ndvi_patterns()
        if ndvi_analysis and ndvi_analysis['nodes_with_data'] > 0:
            print("NDVI PATTERNS:")
            print(f"  Nodes with NDVI data: {ndvi_analysis['nodes_with_data']}")
            print(f"  Mean NDVI: {ndvi_analysis.get('mean_ndvi', 'N/A'):.3f}")
            print(f"  Post-deforestation land use:")
            print(f"    Stable agriculture/vegetation: {ndvi_analysis['stable_agriculture']} nodes")
            print(f"    Recovering land: {ndvi_analysis['recovering_land']} nodes")
            print(f"    Abandoned/bare land: {ndvi_analysis['abandoned_land']} nodes")
            print()
        
        # Neighbor similarity
        similarities = self.analyze_neighbor_similarity()
        if similarities:
            correlations = [s['correlation'] for s in similarities]
            print("NEIGHBOR SIMILARITY:")
            print(f"  Neighbor pairs analyzed: {len(similarities)}")
            print(f"  Mean NDVI correlation: {np.mean(correlations):.3f}")
            print(f"  Correlation range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
            print()
        
        # Soil patterns
        soil_analysis = self.analyze_soil_patterns()
        if soil_analysis and soil_analysis['statistics']:
            print("SOIL PROPERTIES:")
            for prop, stats in soil_analysis['statistics'].items():
                prop_name = prop.replace('_', ' ').title()
                print(f"  {prop_name}:")
                print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            if soil_analysis['correlations']:
                print(f"\n  Correlations with Tree Cover:")
                for corr_name, corr_val in soil_analysis['correlations'].items():
                    prop_name = corr_name.replace('_vs_tree_cover', '').replace('_', ' ').title()
                    print(f"    {prop_name}: {corr_val:.3f}")
            print()
    
    def plot_spatial_map(self, save_path=None):
        """Plot spatial map of deforestation nodes colored by year"""
        if not self.deforested_nodes:
            print("No deforested nodes to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get data
        lons = [n.lon for n in self.deforested_nodes]
        lats = [n.lat for n in self.deforested_nodes]
        years = [n.deforestation_year if n.deforestation_year else 2000 for n in self.deforested_nodes]
        
        # Create scatter plot
        scatter = ax.scatter(lons, lats, c=years, cmap='YlOrRd', 
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deforestation Year', fontsize=12)
        
        # Mark seed point
        ax.scatter([self.seed_lon], [self.seed_lat], c='blue', s=300, 
                  marker='*', edgecolors='black', linewidth=2, 
                  label='Seed Point', zorder=5)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Deforestation Spread Pattern: {self.name}\n({len(self.deforested_nodes)} nodes)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spatial map saved to {save_path}")
        
        plt.show()
    
    def plot_ndvi_timeseries(self, num_nodes=5, save_path=None):
        """Plot NDVI time series for selected nodes"""
        nodes_with_data = [n for n in self.deforested_nodes if n.ndvi_data]
        
        if not nodes_with_data:
            print("No NDVI data available")
            return
        
        # Select nodes with different deforestation years if possible
        selected_nodes = sorted(nodes_with_data, key=lambda n: n.deforestation_year or 2000)[:num_nodes]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_nodes)))
        
        for node, color in zip(selected_nodes, colors):
            dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in node.ndvi_dates]
            ax.plot(dates, node.ndvi_data, label=f'Node {node.node_id} (deforested {node.deforestation_year})',
                   color=color, alpha=0.7, linewidth=1.5)
            
            # Mark deforestation year
            if node.deforestation_year:
                ax.axvline(x=datetime.datetime(node.deforestation_year, 1, 1), 
                          color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('NDVI', fontsize=12)
        ax.set_title(f'NDVI Time Series: {self.name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines for classification thresholds
        ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.3, label='Forest threshold')
        ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.3, label='Agriculture threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"NDVI time series saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_histogram(self, save_path=None):
        """Plot histogram of deforestation years"""
        temporal = self.analyze_temporal_spread()
        
        if not temporal:
            print("No temporal data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        years = sorted(temporal['year_counts'].keys())
        counts = [temporal['year_counts'][y] for y in years]
        
        ax.bar(years, counts, color='orangered', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Nodes Deforested', fontsize=12)
        ax.set_title(f'Temporal Distribution of Deforestation: {self.name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal histogram saved to {save_path}")
        
        plt.show()
    
    def plot_growth_animation(self, save_path=None):
        """Create animated visualization of deforestation spread over time"""
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
        
        if not self.deforested_nodes:
            print("No deforested nodes to animate")
            return
        
        # Sort nodes by deforestation year
        sorted_nodes = sorted([n for n in self.deforested_nodes if n.deforestation_year], 
                             key=lambda n: n.deforestation_year)
        
        if not sorted_nodes:
            print("No nodes with deforestation years")
            return
        
        # Get year range
        years = [n.deforestation_year for n in sorted_nodes]
        min_year = min(years)
        max_year = max(years)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(year):
            ax.clear()
            
            # Get nodes deforested up to this year
            nodes_by_year = [n for n in sorted_nodes if n.deforestation_year <= year]
            
            if nodes_by_year:
                lons = [n.lon for n in nodes_by_year]
                lats = [n.lat for n in nodes_by_year]
                node_years = [n.deforestation_year for n in nodes_by_year]
                
                # Plot with color gradient
                scatter = ax.scatter(lons, lats, c=node_years, cmap='YlOrRd',
                                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                                   vmin=min_year, vmax=max_year)
            
            # Mark seed point
            ax.scatter([self.seed_lon], [self.seed_lat], c='blue', s=300,
                      marker='*', edgecolors='black', linewidth=2, zorder=5)
            
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'{self.name} - Deforestation Spread Through {year}\n'
                        f'{len(nodes_by_year)} nodes deforested', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Keep consistent axis limits
            all_lons = [n.lon for n in sorted_nodes]
            all_lats = [n.lat for n in sorted_nodes]
            ax.set_xlim(min(all_lons) - 0.01, max(all_lons) + 0.01)
            ax.set_ylim(min(all_lats) - 0.01, max(all_lats) + 0.01)
        
        anim = animation.FuncAnimation(fig, update, frames=range(min_year, max_year + 1),
                                      interval=500, repeat=True)
        
        if save_path:
            writer = PillowWriter(fps=2)
            anim.save(save_path, writer=writer)
            print(f"Growth animation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_ndvi_heatmap(self, years=None, save_path=None):
        """Plot heatmap showing NDVI values across nodes at different time points"""
        if not self.deforested_nodes:
            print("No deforested nodes for heatmap")
            return
        
        # Filter nodes with NDVI data
        nodes_with_data = [n for n in self.deforested_nodes if n.ndvi_data]
        
        if not nodes_with_data:
            print("No NDVI data available for heatmap")
            return
        
        # Default to key years if not specified
        if years is None:
            temporal = self.analyze_temporal_spread()
            if temporal:
                first_year = temporal['first_year']
                last_year = temporal['last_year']
                # Select 4-6 representative years
                years = [first_year]
                if last_year - first_year > 10:
                    years.extend([first_year + 5, first_year + 10, first_year + 15, last_year])
                else:
                    years.append(last_year)
                years = [y for y in years if y <= 2024]
            else:
                years = [2005, 2010, 2015, 2020]
        
        # Create subplots for each year
        n_years = len(years)
        fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5))
        
        if n_years == 1:
            axes = [axes]
        
        for ax, year in zip(axes, years):
            # Get NDVI values for this year
            lons = []
            lats = []
            ndvi_values = []
            
            for node in nodes_with_data:
                ndvi = node.get_ndvi_at_year(year)
                if ndvi is not None:
                    lons.append(node.lon)
                    lats.append(node.lat)
                    ndvi_values.append(ndvi)
            
            if ndvi_values:
                # Create scatter plot with NDVI coloring
                scatter = ax.scatter(lons, lats, c=ndvi_values, cmap='RdYlGn',
                                   s=150, alpha=0.8, edgecolors='black', linewidth=0.5,
                                   vmin=0, vmax=0.8)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('NDVI', fontsize=10)
            
            # Mark seed point
            ax.scatter([self.seed_lon], [self.seed_lat], c='blue', s=200,
                      marker='*', edgecolors='black', linewidth=2, zorder=5)
            
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_title(f'Year {year}\n{len(ndvi_values)} nodes', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'NDVI Heatmap Over Time: {self.name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"NDVI heatmap saved to {save_path}")
        
        plt.show()
    
    def export_data(self, filename):
        """Export node data to JSON file"""
        data = {
            'region_name': self.name,
            'seed_coordinates': {'lon': self.seed_lon, 'lat': self.seed_lat},
            'total_nodes': len(self.deforested_nodes),
            'nodes': []
        }
        
        for node in self.deforested_nodes:
            node_data = {
                'id': node.node_id,
                'lon': node.lon,
                'lat': node.lat,
                'deforestation_year': node.deforestation_year,
                'tree_cover_2000': node.tree_cover_2000,
                'num_neighbors': len(node.neighbors),
                'ndvi_stats': node.compute_ndvi_statistics(),
                'soil_properties': {
                    'soil_organic_carbon': node.soil_organic_carbon,
                    'clay_content': node.clay_content,
                    'silt_content': node.silt_content,
                    'sand_content': node.sand_content,
                    'soil_ph': node.soil_ph,
                    'cation_exchange_capacity': node.cation_exchange_capacity
                }
            }
            data['nodes'].append(node_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {filename}")


def main():
    """Main function to process all seed coordinates"""
    
    # Seed coordinates from brazil_example.py
    coordinates = [
        (-62.527222, -9.235, "Rondônia 1"),
        # (-63.279167, -10.252222, "Rondônia 2"),
        # (-54.679444, -2.519722, "Pará"),
        # (-60.182500, -10.244722, "Rondônia 3"),
        # (-6`0.525833, -10.113889, "Rondônia 4"),
        # (-60.678056, -9.991389, "Rondônia 5"),
        # (-55.026389, -1.524722, "Pará 2")
    ]
    
    trackers = []
    
    # Process each coordinate
    for i, (lon, lat, name) in enumerate(coordinates):
        print(f"\n{'#'*60}")
        print(f"Processing region {i+1}/{len(coordinates)}: {name}")
        print(f"{'#'*60}")
        
        tracker = DeforestationTracker(lon, lat, max_radius_km=3, name=name)
        tracker.explore_region(fetch_ndvi=True)
        tracker.generate_report()
        
        # Generate visualizations
        tracker.plot_spatial_map(save_path=f"{name.replace(' ', '_')}_spatial.png")
        tracker.plot_temporal_histogram(save_path=f"{name.replace(' ', '_')}_temporal.png")
        tracker.plot_ndvi_timeseries(num_nodes=5, save_path=f"{name.replace(' ', '_')}_ndvi.png")
        tracker.plot_ndvi_heatmap(save_path=f"{name.replace(' ', '_')}_heatmap.png")
        tracker.plot_growth_animation(save_path=f"{name.replace(' ', '_')}_animation.gif")
        
        # Export data
        tracker.export_data(f"{name.replace(' ', '_')}_data.json")
        
        trackers.append(tracker)
    
    print(f"\n{'='*60}")
    print("ALL REGIONS PROCESSED")
    print(f"{'='*60}")
    print(f"Total regions analyzed: {len(trackers)}")
    print(f"Total deforested nodes found: {sum(len(t.deforested_nodes) for t in trackers)}")
    

if __name__ == "__main__":
    main()

