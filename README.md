# CO₂ Capture and Storage (CCS) Optimization Tool

This repository contains a **CCS Optimization Tool**, a Python-based solution utilizing **Streamlit** for interactive visualization and **scipy.optimize** for mathematical optimization. The tool is designed to efficiently match CO₂ emission sources with storage sinks and optimize infrastructure planning for carbon capture and transportation.

## Features
- **Interactive Web Interface:** Streamlit-powered GUI for user interaction.
- **Data Processing & Clustering:** Uses DBSCAN clustering to group CO₂ sources.
- **Pipeline Optimization:** Determines optimal pipeline networks based on cost and distance.
- **Cost & Environmental Impact Analysis:** Estimates costs for capture, transport, and storage.
- **Geospatial Visualization:** Uses **Folium** for mapping CO₂ sources, sinks, and pipelines.

## Installation
### **Requirements**
Ensure you have Python installed (Python 3.8 or later). Install the necessary dependencies using:
```sh
pip install -r requirements.txt
```
## **Usage**
Clone the repository:
```
git clone https://github.com/yourusername/ccs-optimization.git
```
Navigate into the directory:
```
cd ccs-optimization
```
Run the Streamlit application:
```
streamlit run CCC-NetOptimizer.py
```
Upload an Excel file containing:
## **Input file**
- CO₂ Sources (Latitude, Longitude, Emission, Type, etc.)
- CO₂ Sinks (Capacity, Type, Location)
- Pipeline Data (Existing pipeline networks)
Adjust the eps clustering parameter and analyze the results.
