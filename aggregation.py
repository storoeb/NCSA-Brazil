import ee
import geemap

ee.Initialize(project='fluent-cosine-473703-g7')

# Load Brazil boundary
brazil = (
    ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
    .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
)

# Soil layers to visualize with units and descriptions
soil_layers = {
    "ocd_mean": {
        "band": "ocd_0-5cm_mean",
        "label": "Organic Carbon Density",
        "unit": "g/kg"
    },
    "phh2o_mean": {
        "band": "phh2o_0-5cm_mean",
        "label": "Soil pH (H2O)",
        "unit": "pH"
    },
    "sand_mean": {
        "band": "sand_0-5cm_mean",
        "label": "Sand Fraction",
        "unit": "g/kg"
    },
    "clay_mean": {
        "band": "clay_0-5cm_mean",
        "label": "Clay Fraction",
        "unit": "g/kg"
    },
    "cec_mean": {
        "band": "cec_0-5cm_mean",
        "label": "Cation Exchange Capacity",
        "unit": "mmol(c)/kg"
    },
}

def get_palette(key):
    """Get the color palette for a given soil layer key"""
    if "phh2o" in key:
        return ["#67001f","#d6604d","#f7f7f7","#4393c3","#053061"]
    elif "sand" in key:
        return ["#ffffcc","#fd8d3c","#e31a1c","#800026"]
    elif "clay" in key:
        return ["#f7fcf0","#74c476","#00441b"]
    elif "p_" in key:
        return ["#fef0d9","#fdcc8a","#fc8d59","#d7301f","#7f0000"]
    else:
        return ["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]

# Create a map with layer control disabled initially
Map = geemap.Map(center=[-10, -55], zoom=4, add_layer_control=False)
Map.clear_layers()  # Clear any default layers
Map.add_basemap("HYBRID")

# Store stats for legend creation
layer_stats = {}

# Loop through each layer and auto-scale
for key, layer_info in soil_layers.items():
    band = layer_info["band"]
    img = ee.Image(f"projects/soilgrids-isric/{key}").select(band).clip(brazil)
    
    # Dynamically compute min/max over Brazil
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=brazil.geometry(),
        scale=5000,        # coarser scale for speed
        maxPixels=1e13
    ).getInfo()

    # Extract stats
    min_val = stats[f"{band}_min"]
    max_val = stats[f"{band}_max"]
    
    # pH values are stored multiplied by 10, so divide by 10 for display
    if "phh2o" in key:
        min_val = min_val / 10.0
        max_val = max_val / 10.0
    
    # Store stats for legend
    layer_stats[key] = {"min": min_val, "max": max_val}

    print(f"{key}: min={min_val:.2f}, max={max_val:.2f}")

    # Get palette for this layer
    palette = get_palette(key)
    
    # For pH, use the original values for visualization but scaled values for display
    if "phh2o" in key:
        vis_params = {"min": min_val * 10, "max": max_val * 10, "palette": palette}
    else:
        vis_params = {"min": min_val, "max": max_val, "palette": palette}

    layer_name = layer_info["label"]
    Map.addLayer(img, vis_params, layer_name)

# Add Brazil boundary
Map.addLayer(brazil, {"color": "black"}, "Brazil Boundary")

# Add layer control back
Map.add_layer_control()

# Create HTML legend content for all layers
legend_html = """
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 280px; height: auto; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius: 5px; padding: 10px">
     
<h4 style="margin-top:0">Soil Layer Legends</h4>
"""

# Add legend for each layer
for key, layer_info in soil_layers.items():
    legend_html += f"<p style='margin: 10px 0 5px 0; font-weight: bold;'>{layer_info['label']} ({layer_info['unit']})</p>"
    
    # Get the palette for this layer
    palette = get_palette(key)
    
    # Create gradient bar
    gradient = "linear-gradient(to right, " + ", ".join(palette) + ")"
    legend_html += f'<div style="background: {gradient}; height: 20px; width: 100%; border: 1px solid #ccc;"></div>'
    
    # Add min/max labels from stored stats
    min_val = layer_stats[key]["min"]
    max_val = layer_stats[key]["max"]
    
    legend_html += f'<div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 15px;">'
    legend_html += f'<span>{min_val:.1f}</span><span>{max_val:.1f}</span></div>'

legend_html += "</div>"

# Save map to HTML first
Map.to_html("Brazil_SoilLayers_AutoScaled.html", title="Auto-scaled Soil Properties Across Brazil")

# Read the HTML file and inject the legend
with open("Brazil_SoilLayers_AutoScaled.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Insert the legend before the closing body tag
html_content = html_content.replace("</body>", legend_html + "\n</body>")

# Write the modified HTML back
with open("Brazil_SoilLayers_AutoScaled.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Map saved as Brazil_SoilLayers_AutoScaled.html")
