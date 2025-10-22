import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='fluent-cosine-473703-g7')

# ---------------------------------------------------------------------
# 1️⃣  Load Brazil municipalities (GAUL Level 2)
# ---------------------------------------------------------------------
municipios = (
    ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
    .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
)

# ---------------------------------------------------------------------
# 2️⃣  Load Amazon biome boundary (MapBiomas public asset)
# ---------------------------------------------------------------------
amazon = (
    ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/biomas-2019")
    .filter(ee.Filter.eq("Bioma", "Amazônia"))
)

# ---------------------------------------------------------------------
# 3️⃣  Filter municipalities intersecting the Amazon biome
# ---------------------------------------------------------------------
municipios_amazon = municipios.filterBounds(amazon.geometry())

# ---------------------------------------------------------------------
# 4️⃣  Load soil dataset (SoilGrids - Organic Carbon, 0–5 cm)
# ---------------------------------------------------------------------
# Using OpenLandMap Soil Organic Carbon dataset
soil = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select("b0")

# ---------------------------------------------------------------------
# 5️⃣  Compute average soil carbon for each municipality
# ---------------------------------------------------------------------
def add_mean_soil(feature):
    mean_value = soil.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=250,
        maxPixels=1e13
    ).get("b0")
    return feature.set("mean_OC", mean_value)

results = municipios_amazon.map(add_mean_soil)

# ---------------------------------------------------------------------
# 6️⃣  Style the municipalities by mean soil value
# ---------------------------------------------------------------------
# Define a color ramp for visualization (values in g/kg x 10)
vis_params = {
    "min": 0,
    "max": 120,
    "palette": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
}

# Convert feature property to an image for visualization
soil_map = results.reduceToImage(properties=["mean_OC"], reducer=ee.Reducer.first())

# ---------------------------------------------------------------------
# 7️⃣  Create an interactive map
# ---------------------------------------------------------------------
Map = geemap.Map(center=[-5, -60], zoom=4)
Map.add_basemap("SATELLITE")
Map.addLayer(soil, vis_params, "Soil Carbon (raw)")
Map.addLayer(soil_map, vis_params, "Mean Soil OC by Municipality")
Map.addLayer(amazon, {"color": "green"}, "Amazon Biome Boundary")
Map.addLayerControl()

# Save the map to an HTML file
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
map_filename = f"amazon_soil_carbon_map_{timestamp}.html"
Map.to_html(map_filename)
print(f"Map saved to: {map_filename}")

# # ---------------------------------------------------------------------
# # 8️⃣  Export results to Google Drive
# # ---------------------------------------------------------------------
# task = ee.batch.Export.table.toDrive(
#     collection=results,
#     description="Amazon_Soil_OC_By_Municipality",
#     folder="EarthEngineExports",
#     fileNamePrefix="Amazon_Soil_OC_By_Municipality",
#     fileFormat="CSV"
# )
# task.start()

# print("✅ Export started! Check your Google Drive in a few minutes.")
