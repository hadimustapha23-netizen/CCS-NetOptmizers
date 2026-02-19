import streamlit as st 
import pandas as pd 
import geopandas as gpd 
import scipy.optimize as opt
import folium 
from streamlit_folium import folium_static 
from sklearn.cluster import DBSCAN 
from shapely.ops import nearest_points 
import numpy as np 
from scipy.sparse.csgraph import minimum_spanning_tree 
from scipy.spatial.distance import pdist, squareform 
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point
import math
from scipy.optimize import fsolve

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Page 1", "Page 2", "Page 3"])

    if page == "Page 1":
        page1()
    elif page == "Page 2":
        page2()
    elif page == "Page 3":
        page3()

def page1():
    st.title("Project Information")
    st.write("This project aims to build networks of CO2 sources and sinks.")
    
def page2():
    st.title("Upload Excel File")
    file = st.file_uploader("Upload an Excel file", type=['xlsx'])
    ineps = st.number_input("Enter the eps:", min_value=0.0, step=0.1)
    if file is not None:
        df_sources = pd.read_excel(file, sheet_name='CO2 sources')  
        # Normalise column names
df_sources.columns = df_sources.columns.str.strip()

# Allow common variants
emission_col_candidates = ["Emission", "Emissions", "CO2 Emission", "CO2 Emissions", "emission", "emissions"]
emission_col = next((c for c in emission_col_candidates if c in df_sources.columns), None)

if emission_col is None:
    st.error(
        "Missing emissions column in 'CO2 sources' sheet. "
        "Expected a column named 'Emission' (or a common variant like 'Emissions'). "
        f"Found columns: {list(df_sources.columns)}"
    )
    st.stop()

df_sources["Total Emitted"] = df_sources[emission_col]

        df_sinks = pd.read_excel(file, sheet_name='CO2 sinks')  
        df_pipelines = pd.read_excel(file, sheet_name='pipeline')
        return df_sources, df_sinks, df_pipelines, ineps 
    return None, None, ineps
def page3():
    df_sources, df_sinks, df_pipelines, ineps = page2()

    # Ensure df_sources and df_sinks exist and are not empty
    if df_sources.empty:
        raise ValueError("df_sources is empty")
    if df_sinks.empty:
        raise ValueError("df_sinks is empty")
    if df_pipelines.empty:
        raise ValueError("df_pipelines is empty")
    
    # Calculate total emitted and total captured
    df_sources['Total Emitted'] = df_sources['Emission']
    df_sources['Captured Coefficient'] = df_sources['Type'].map({
        'cement': 0.8075,
        'NG': 0.9,
        'PC': 0.9,
        'IS': 0.66,
        'GP': 0.58,
        'OR': 0.744
    })
    df_sources['Total Captured'] = df_sources['Emission'] * df_sources['Captured Coefficient']
    # Calculate capture cost
    capture_cost_params = {
        'cement': {'COC': 132},
        'PC':  {'COC': 94},
        'NG':  {'COC': 166},
        'IS':  {'COC': 115},
        'GP':  {'COC': 29},
        'OR':  {'COC': 115}
    }
    storage_cost_params = {
        "WT": {'a_1': 31226, 'a_2': 0.0000857, 'DC_a_1': 43986, 'DC_a_2': 0.00034},
        "ST": {'a_1': 37040, 'a_2': 0.0000354, 'DC_a_1': 44041, 'DC_a_2': 0.00035},
        "SL": {'a_1': 39876, 'a_2': 0.0000345, 'DC_a_1': 44041, 'DC_a_2': 0.00035},
        "MC": {'a_1': 39876, 'a_2': 0.0000345, 'DC_a_1': 42493, 'DC_a_2': 0.00035},
        "RM": {'a_1': 29611, 'a_2': 0.0000792, 'DC_a_1': 80086, 'DC_a_2': 0.00027},
        "C": {'a_1': 38931, 'a_2': 0.0000639, 'DC_a_1': 70123, 'DC_a_2': 0.00032}      
    }
    def calculate_capture_cost(row):
        params = capture_cost_params.get(row['Type'], None)
        if params:
            return params['COC'] * (row['Total Captured'])/1e6   
        else:
            return 0
            
    def calculate_storage_cost(row, total_captured_Mt):
        storage_params = storage_cost_params.get(row['Type'], None)
        if storage_params:
            storage_cost_equipping = storage_params['a_1'] * math.exp(storage_params['a_2'] * row['d_m'] * 3.28084)*1.67/1e6
            storage_cost_DC = storage_params['DC_a_1'] * math.exp(storage_params['DC_a_2'] * row['d_m'] * 3.28084)*1.67/1e6
            
            n_wells = math.ceil(total_captured_Mt / 1.2)
            storage_powerMWh=((row['ro_kgm3']*9.81*row['d_m'])+(0.05*row['d_m']*row['ro_kgm3']*7.5*7.5/2*1.05))*total_captured_Mt*31.7/row['ro_kgm3']/1000000
            storage_powerMWh
            storage_cost_compressor = (8.35 * storage_powerMWh + 0.49) * 1.67
            EI_inject_compression_energy = 882 * storage_powerMWh/1000
            if n_wells >= 21: 
                return (storage_cost_equipping + storage_cost_DC + storage_cost_compressor), n_wells, EI_inject_compression_energy
            else: 
                return (storage_cost_equipping * (21 / n_wells)**0.5) + storage_cost_compressor + storage_cost_DC, n_wells, EI_inject_compression_energy
        else:
            return 0
    
    def routes_with_capacity(total_captured, route_distance):
        pipelines = []
        remaining_CO2 = total_captured
        route_number = 0
          
        while remaining_CO2 > 21000000:
                
            remaining_CO2 -= 21000000
            route_number = route_number + 1
            pipelines.append([route_number, 21000000])
        
        # For the last pipeline segment, if CO₂ is less than 10 Mt, add the remaining amount
        if remaining_CO2 > 0:
            route_number = route_number + 1
            pipelines.append([route_number, remaining_CO2])  
        return route_number  

    def calculate_pipeline_cost(D, L):
        # Convert D to inches and L to km
        D_inches = D * 39.3701
        L_km = L / 1000
        
        # Pipeline cost formula
        cost = ((10**3.112 * L_km**0.901 * D_inches**1.59) + (10**4.487 * L_km**0.82 * D_inches**0.94) + (10**3.95 * L_km**1.049 * D_inches**0.403) + (10**4.39 * L_km**0.783 * D_inches**0.791)) * 1.67 / 1e6
        
        return cost

    def round_to_nominal_pipe_size(D):
        nominal_sizes = [0.152, 0.203, 0.254, 0.305, 0.356, 0.406, 0.457, 0.508, 0.610, 0.762, 0.914, 1.016]  # Example: 6 to 24 inches
        return min(nominal_sizes, key=lambda x: abs(x - D))

    def friction_factor(Re, pipe_roughness, D):
        #Compute friction factor using the Colebrook equation for turbulent flow 
        if Re < 2000:  # Laminar flow
            return 64 / Re
        else:  # Turbulent flow (Colebrook approximation)
            return (1 / (-1.8 * np.log10((pipe_roughness / (3.7 * D))**1.11 + 6.9 / Re)))**2

    # Darcy-Weisbach pressure drop equation
    def calculate_pressure_drop(mass_flow_rate, D, L, rho, mu, pipe_roughness):
        if D < 0.01:  # Ensure minimum pipe diameter
            return float('inf')

        A = np.pi * (D**2) / 4  # Pipe cross-sectional area
        v = mass_flow_rate / (rho * A)  # Velocity

        Re = (rho * v * D) / mu  # Reynolds number
        f = friction_factor(Re, pipe_roughness, D)  # Compute friction factor

        delta_P = (f * L * rho * v**2) / (2 * D)  # Pressure drop using Darcy-Weisbach equation
        return max(delta_P, 1e4)  # Ensure at least small pressure drop to avoid instability

    # Simplified pipeline diameter estimation (based on empirical velocity constraints)
    def calculate_pipeline_diameter(mass_flow_rate, pressure_in, length, temp_avg, pipe_roughness=0.0457e-3, efficiency=0.85):
        R = 8.314
        M_CO2 = 44.01e-3
        velocity = 2  # Assume a safe transport velocity of 2 m/s
        mu = 1.48e-5  # Dynamic viscosity of CO2

        # Approximate initial pipeline diameter using mass flow rate and velocity
        D = max(np.sqrt((4 * mass_flow_rate) / (np.pi * velocity * 1000)), 0.1)

        # Approximate CO2 density
        rho = pressure_in * M_CO2 / (R * temp_avg)

        # Compute pressure drop
        delta_P = calculate_pressure_drop(mass_flow_rate, D, length, rho, mu, pipe_roughness)
        pressure_out = max(pressure_in - delta_P, 7.38e6)  # Ensure min pressure is met

        # Prevent pressure_out > pressure_in error
        if pressure_out >= pressure_in:
            pressure_out = 0.9 * pressure_in

        # Compute booster power using a more stable formula
        k = 1.28  # CO2 specific heat ratio
        compression_ratio = (pressure_out / pressure_in)
        if compression_ratio < 1.1:  # Ensure meaningful compression
            compression_ratio = 1.1

        W = ((R * temp_avg) / efficiency) * (k / (k - 1)) * ((compression_ratio-1)**((k - 1)/k)) * mass_flow_rate / 1e6
        D = round_to_nominal_pipe_size(D)

        return float(D), float(W), float(pressure_out)
   
   
    df_sources['Capture Cost'] = df_sources.apply(calculate_capture_cost, axis=1)
    

    gdf_sources = gpd.GeoDataFrame(df_sources, geometry=gpd.points_from_xy(df_sources.Longitude, df_sources.Latitude))
    gdf_sinks = gpd.GeoDataFrame(df_sinks, geometry=gpd.points_from_xy(df_sinks.Longitude, df_sinks.Latitude))
    gdf_pipelines = gpd.GeoDataFrame(df_pipelines, geometry=gpd.points_from_xy(df_pipelines.Longitude, df_pipelines.Latitude))

    grouped_pipelines = gdf_pipelines.groupby('Group').apply(lambda x: x.sort_values(by='Pipeline ID')).reset_index(drop=True)
    
    average_latitude = df_sources['Latitude'].mean()
    average_longitude = df_sources['Longitude'].mean()
    m = folium.Map(
    location=[average_latitude, average_longitude], 
    zoom_start=4.4, 
    tiles='CartoDB positron', 
    attr='&copy; <a href="https://carto.com/attributions">CARTO</a>'
)
    
    # Connect points within each group in order
    for group, group_data in grouped_pipelines.groupby('Group'):
        points = group_data['geometry'].to_list()
        for i in range(len(points) - 1):
            line = LineString([points[i].coords[0], points[i+1].coords[0]])
            folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]), (line.coords[1][1], line.coords[1][0])], color="black", dash_array="5, 5").add_to(m)     
        
    source_colors = {
    'cement': 'darkblue',
    'NG': 'darkred',
    'PC': 'purple',
    'IS': 'orange',
    'GP': 'gray',
    'OR': 'darkgreen'
    }

    # Add CO₂ sources to the map with color coding
    for idx, row in gdf_sources.iterrows():
        source_type = row['Type']
        color = source_colors.get(source_type, 'gray')  # Default to gray if type not in dictionary
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            popup=f"Source: {row['CO2 sources']}, Type: {source_type}"
        ).add_to(m)
    for idx, row in gdf_sinks.iterrows():
        folium.RegularPolygonMarker(
            location=[row['Latitude'], row['Longitude']],
            number_of_sides=3,  # Triangle shape
            radius=3,  # Adjust size as needed
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            popup=f"Sink: {row['CO2 sinks']}, Capacity: {row['Cap']} tCO2"
        ).add_to(m)

    # Convert the sources to a 2D array for clustering
    coordinates = gdf_sources.geometry.apply(lambda geom: np.array(geom.coords[0])).to_list()

    # Use DBSCAN to cluster the sources for eps=3.4
    eps = ineps
    db = DBSCAN(eps=eps, min_samples=2).fit(coordinates)
    labels = db.labels_

    plot_data = {
        'eps': [],
        'Total Captured tCO2': [],
        'Total Emitted tCO2': [],
        'Environmental impact': [],
        'Total Distance':[],
        'Transportation Cost MUSD': [],
        'Storage Cost MUSD':[],
        'Total Capture Cost MUSD': [],
        'Total Cost MUSD': [],
        'EI_of_transp': [],
        'EI_inject': []
      
        
    }

   
    # Add the cluster labels to the sources GeoDataFrame
    gdf_sources['cluster'] = labels
    sink_data = {}
    total_distance = 0   
    distance = 0
    EI_inject = 0
    pipeline_costs = []
    hub_pipeline_cost = 0
    hub_pbf_cost = 0
    sink_pipeline_cost = 0
    sink_pbf_cost = 0
    # Iterate over each cluster
    for cluster in set(labels):
        if cluster != -1:  # Ignore noise (points not in any cluster)
            sources_in_cluster = gdf_sources[gdf_sources['cluster'] == cluster]
            #cluster_counts = gdf_sources['cluster'].value_counts()
            #cluster_counts
            total_emitted = sources_in_cluster['Total Emitted'].sum()
            total_captured = sources_in_cluster['Total Captured'].sum()
            #total_captured
            total_capture_cost = sources_in_cluster['Capture Cost'].sum()
            distance = 0
            # Compute the distance matrix between all sources in the cluster
            dist_matrix = squareform(pdist(sources_in_cluster[['Latitude', 'Longitude']].values))

            # Compute the minimum spanning tree
            mst = minimum_spanning_tree(dist_matrix).toarray()
            cumulative_mass_flow = 0 
            pressure_in = 15e6  
            processed_sources = set()
            current_cumulative_mass_flow = 0
            EI_of_transportation_source = 0
            EI_of_transportation_sink = 0


            

            # Draw a line between each pair of sources connected by the MST
            for idx1 in range(mst.shape[0]):
                for idx2 in range(mst.shape[1]):
                    if mst[idx1, idx2] > 0:
                        
                        source1 = sources_in_cluster.iloc[idx1]
                        source2 = sources_in_cluster.iloc[idx2]
                        line = LineString([source1.geometry, source2.geometry])  # Convert to meters
                        folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]),
                                                   (line.coords[1][1], line.coords[1][0])], color="green", weight=1).add_to(m)
                        segment_length = LineString([source1.geometry, source2.geometry]).length * 100000 
                        
                        
                        route_number = routes_with_capacity(total_captured, line.length*100)
                        total_distance += route_number*line.length
                        distance += route_number*line.length 
                            

                        if source1.name not in processed_sources:
                            cumulative_mass_flow += source1['Total Captured']
                            processed_sources.add(source1.name)
                        if source2.name not in processed_sources:
                            cumulative_mass_flow += source2['Total Captured']
                            processed_sources.add(source2.name)
                        
                        if route_number > 1:
                            current_cumulative_mass_flow = cumulative_mass_flow / route_number
                        else:
                            current_cumulative_mass_flow = cumulative_mass_flow

                        
                        diameter, pbf_power, pressure_out = calculate_pipeline_diameter(current_cumulative_mass_flow*3.171e-5, pressure_in, segment_length, 285.15)
                        pipeline_cost = calculate_pipeline_cost(diameter, segment_length)
                        hub_pipeline_cost += pipeline_cost
                        pipeline_costs.append(pipeline_cost)
                        pbf_cost = (8.35 * pbf_power + 0.49) * 1.67
                        hub_pbf_cost += pbf_cost
                        pressure_in = pressure_out
                        if pressure_in < 7.38e6:
                            pressure_in = 15e6

                        
                        EI_pbf_energy = 882 * pbf_power / 3600 / 1000
                        EI_pipeline_fugitive = 1.4 * segment_length/1000
                        EI_of_transportation_source +=(EI_pbf_energy + EI_pipeline_fugitive)*route_number
                         
                                
                                              
                        
                    
            
            # Find the nearest sink to the cluster that can handle the emissions
            sorted_sinks = gdf_sinks.copy()
            sorted_sinks['distance'] = sorted_sinks.distance(sources_in_cluster.unary_union)
            sorted_sinks = sorted_sinks.sort_values(by='distance')

            for _, sink_row in sorted_sinks.iterrows():
                sink_name = sink_row['CO2 sinks']
                sink_capacity = sink_row['Cap']
                
                if sink_name in sink_data:
                    current_total_captured = sink_data[sink_name]['Total Captured']    
                else:
                    current_total_captured = 0
                    
                if current_total_captured + total_captured <= sink_capacity:
                    nearest_sink = sink_row.geometry
                    nearest_sink_name = sink_name
                    nearest_sink_capacity = sink_capacity
                    break

            distances_to_sink = sources_in_cluster.distance(nearest_sink)
            nearest_source = sources_in_cluster.loc[distances_to_sink.idxmin()].geometry
            line = LineString([nearest_sink, nearest_source])
            folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]),
                                       (line.coords[1][1], line.coords[1][0])], color="blue", weight=1).add_to(m)
            route_number = routes_with_capacity(total_captured, line.length*100)

            total_distance += route_number*line.length
            distance += route_number*line.length

            
            diameter, sink_pbf_power, pressure_out = calculate_pipeline_diameter(total_captured*3.171e-5, pressure_in, line.length*100000, 285.15)
            sink_pipeline_cost = calculate_pipeline_cost(diameter, line.length*100000)              
            sink_pbf_cost = (8.35 * sink_pbf_power + 0.49) * 1.67       
            pressure_in = pressure_out
            if pressure_in < 7.38e6:
                pressure_in = 15e6

           
            EI_pbf_energy_sink = 882 * sink_pbf_power / 3600 / 1000
            EI_pipeline_fugitive_sink = 1.4 * line.length*100
            EI_of_transportation_sink += (EI_pbf_energy_sink + EI_pipeline_fugitive_sink)*route_number
            

            EI_of_transp = EI_of_transportation_sink + EI_of_transportation_source

            if nearest_sink_name in sink_data:
                sink_data[nearest_sink_name]['Total Captured'] += total_captured
                sink_data[nearest_sink_name]['Total Emitted'] += total_emitted
                sink_data[nearest_sink_name]['Total Capture Cost'] += total_capture_cost
                sink_data[nearest_sink_name]['Total Distance'] += distance
                sink_data[nearest_sink_name]['Storage Cost'] = 0
                sink_data[nearest_sink_name]['n_wells'] = 0
                sink_data[nearest_sink_name]['EI_of_transp'] += EI_of_transp 
                sink_data[nearest_sink_name]['EI_of_transp_inject'] += EI_of_transp 
                sink_data[nearest_sink_name]['route_number'] += route_number
                sink_data[nearest_sink_name]['EI_inject'] += 0
                sink_data[nearest_sink_name]['Number of Sources'] += len(sources_in_cluster)
                sink_data[nearest_sink_name]['Total Pipeline Cost'] += hub_pipeline_cost*route_number
                sink_data[nearest_sink_name]['Total Pipeline Cost'] += sink_pipeline_cost
                sink_data[nearest_sink_name]['Total pipeline Booster Cost'] += hub_pbf_cost*route_number
                sink_data[nearest_sink_name]['Total pipeline Booster Cost'] += sink_pbf_cost

                
            else:
                sink_data[nearest_sink_name] = {
                    'Total Captured': total_captured,
                    'Total Emitted': total_emitted,
                    'Total Capture Cost': total_capture_cost,
                    'Capacity': nearest_sink_capacity,
                    'Total Distance': distance,
                    'Sink Lifetime': nearest_sink_capacity/total_captured,
                    'Storage Cost': 0,
                    'n_wells': 0,
                    'EI_of_transp': EI_of_transp,
                    'EI_of_transp_inject': EI_of_transp,
                    'route_number': route_number,
                    'EI_inject': 0,
                    'Number of Sources': len(sources_in_cluster),
                    'Total Pipeline Cost': hub_pipeline_cost*route_number+sink_pipeline_cost,
                    'Total pipeline Booster Cost': hub_pbf_cost*route_number+sink_pbf_cost   
                }
    for sink_name, data in sink_data.items():
        # Convert total captured to million tonnes for storage cost calculation
        total_captured_million = data['Total Captured'] / 1e6
        
        # Get sink's characteristics
        sink_row = gdf_sinks[gdf_sinks['CO2 sinks'] == sink_name].iloc[0]
        d_m = sink_row['d_m']
        ro_kgm3 = sink_row['ro kg/m3']
        sink_type = sink_row['Type']

        # Calculate the storage cost using the cumulative captured emissions
        storage_cost, n_wells, EI_inject_compression_energy = calculate_storage_cost(
            {'Type': sink_type, 'd_m': d_m, 'ro_kgm3': ro_kgm3}, total_captured_million
        )

        # Update the sink data dictionary
        data['Storage Cost'] = storage_cost
        data['n_wells'] = n_wells
        data['EI_of_transp_inject'] += EI_inject_compression_energy
        data['EI_inject']= EI_inject_compression_energy

    total_captured_tCO2 = sum(data['Total Captured'] for data in sink_data.values())
    total_emitted_tCO2 = sum(data['Total Emitted'] for data in sink_data.values())
    total_capture_cost_MUSD = sum(data['Total Capture Cost'] for data in sink_data.values())
    transportation_pipeline_cost_MUSD = sum(data['Total Pipeline Cost'] for data in sink_data.values())
    transportation_pbf_MUSD = sum(data['Total pipeline Booster Cost'] for data in sink_data.values())
    environmental_impact_m = sum(data['EI_of_transp_inject'] for data in sink_data.values())
    total_storage_cost_MUSD = sum(data['Storage Cost'] for data in sink_data.values())
    EI_of_transp = sum(data['EI_of_transp'] for data in sink_data.values())
    
    EI_inject = sum(data['EI_inject'] for data in sink_data.values())
    
    plot_data['eps'].append(eps)
    plot_data['Total Captured tCO2'].append(total_captured_tCO2)
    plot_data['Total Emitted tCO2'].append(total_emitted_tCO2)


    plot_data['Total Distance'].append(total_distance * 100)
    plot_data['Environmental impact'].append(environmental_impact_m)
    plot_data['Transportation Cost MUSD']= transportation_pipeline_cost_MUSD+transportation_pbf_MUSD
    plot_data['Storage Cost MUSD'].append(total_storage_cost_MUSD)
    plot_data['Total Capture Cost MUSD'].append(total_capture_cost_MUSD)
    plot_data['Total Cost MUSD'].append(total_capture_cost_MUSD +transportation_pipeline_cost_MUSD + transportation_pbf_MUSD+ total_storage_cost_MUSD)
    plot_data['EI_of_transp'].append(EI_of_transp)
    plot_data['EI_inject'].append(EI_inject)
    

    plot_df = pd.DataFrame(plot_data)
    st.write("### Full Redirection Map")
    # Display the map 
    folium_static(m)
    
    # Prepare sink_data for display
    sink_data_df = pd.DataFrame.from_dict(sink_data, orient='index').reset_index()
    sink_data_df.columns = ['Sink Name', 'Total Captured', 'Total Emitted', 'Total Capture Cost', 'Sink Capacity', 'Total Distance', 'Sink Lifetime', 'Storage Cost', 'n_wells', 'EI_of_transp','EI_of_transp_inject', 'route_number', 'EI_inject', 'Number of Sources', 'Total Pipeline Cost', 'Total pipeline Booster Cost']

    # Display the table
    st.table(sink_data_df)
    # Output the table
    st.write("### Analysis Table")
    st.write(plot_df)




# Create a second map
    
    m2 = folium.Map(
    location=[average_latitude, average_longitude], 
    zoom_start=4.4, 
    tiles='CartoDB positron', 
    attr='&copy; <a href="https://carto.com/attributions">CARTO</a>'
)
    
    # Connect points within each group in order
    for group, group_data in grouped_pipelines.groupby('Group'):
        points = group_data['geometry'].to_list()
        for i in range(len(points) - 1):
            line = LineString([points[i].coords[0], points[i+1].coords[0]])
            folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]), (line.coords[1][1], line.coords[1][0])], color="black", weight=2, dash_array="5, 5").add_to(m2)

            
        
    for idx, row in gdf_sources.iterrows():
        source_type = row['Type']
        color = source_colors.get(source_type, 'gray')  # Default to gray if type not in dictionary
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            popup=f"Source: {row['CO2 sources']}, Type: {source_type}"
        ).add_to(m2)
    for idx, row in gdf_sinks.iterrows():
        folium.RegularPolygonMarker(
            location=[row['Latitude'], row['Longitude']],
            number_of_sides=3,  # Triangle shape
            radius=3,  # Adjust size as needed
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            popup=f"Sink: {row['CO2 sinks']}, Capacity: {row['Cap']} tCO2"
        ).add_to(m2)
    
    eps = ineps
    db = DBSCAN(eps=eps, min_samples=2).fit(coordinates)
    labels = db.labels_

    plot_data2 = {
        'eps': [],
        'Total Captured tCO2': [],
        'Total Emitted tCO2': [],
        'Environmental impact': [],
        'Total Distance':[],
        'Transportation Cost MUSD': [],
        'Storage Cost MUSD':[],
        'Total Capture Cost MUSD': [],
        'Total Cost MUSD': [],
        'EI_of_transp': [],
        'EI_inject': []
        
        
    }
    
    partial_sink_data = {}
    total_distance_partial = 0
    EI_inject = 0
    partial_hub_pipeline_cost = 0
    partial_hub_pbf_cost = 0
    

    for cluster in set(labels):
        if cluster != -1:  # Ignore noise (points not in any cluster)
            sources_in_cluster = gdf_sources[gdf_sources['cluster'] == cluster]
            total_emitted = sources_in_cluster['Total Emitted'].sum()
            total_captured = sources_in_cluster['Total Captured'].sum()
            total_capture_cost = sources_in_cluster['Capture Cost'].sum()
            distance_partial = 0

            # Compute the distance matrix between all sources in the cluster
            dist_matrix = squareform(pdist(sources_in_cluster[['Latitude', 'Longitude']].values))

            # Compute the minimum spanning tree
            mst = minimum_spanning_tree(dist_matrix).toarray()
            cumulative_mass_flow = 0 
            pressure_in = 15e6  
            processed_sources = set()
            current_cumulative_mass_flow = 0
            EI_of_transportation_source = 0
            EI_of_transportation_sink = 0
            # Draw a line between each pair of sources connected by the MST
            for idx1 in range(mst.shape[0]):
                for idx2 in range(mst.shape[1]):
                    if mst[idx1, idx2] > 0:
                        source1 = sources_in_cluster.iloc[idx1]
                        source2 = sources_in_cluster.iloc[idx2]
                        line = LineString([source1.geometry, source2.geometry])
                        folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]),
                                                (line.coords[1][1], line.coords[1][0])], color="green", weight=1).add_to(m2)
                        route_number = routes_with_capacity(total_captured, line.length*100)
                        partial_segment_length = LineString([source1.geometry, source2.geometry]).length * 100000 
                        segment_distance = route_number*line.length
                        distance_partial += segment_distance


                        if source1.name not in processed_sources:
                            cumulative_mass_flow += source1['Total Captured']
                            processed_sources.add(source1.name)
                        if source2.name not in processed_sources:
                            cumulative_mass_flow += source2['Total Captured']
                            processed_sources.add(source2.name)

                        if route_number > 1:
                            current_cumulative_mass_flow = cumulative_mass_flow / route_number
                        else:
                            current_cumulative_mass_flow = cumulative_mass_flow 


                        diameter, pbf_power, pressure_out = calculate_pipeline_diameter(current_cumulative_mass_flow*3.171e-5, pressure_in, partial_segment_length, 285.15)
                        partial_pipeline_cost = calculate_pipeline_cost(diameter, partial_segment_length)              
                        partial_pbf_cost = (8.35 * pbf_power + 0.49) * 1.67      
                        partial_hub_pbf_cost += partial_pbf_cost 
                        partial_hub_pipeline_cost += partial_pipeline_cost
                        pressure_in = pressure_out
                        if pressure_in < 7.38e6:
                            pressure_in = 15e6

                        
                        EI_pbf_energy = 882 * pbf_power / 3600 / 1000
                        EI_pipeline_fugitive = 1.4 * partial_segment_length/1000
                        EI_of_transportation_source += (EI_pbf_energy + EI_pipeline_fugitive)*route_number
                                   

            # Find the nearest sink to the cluster that can handle the emissions
            sorted_sinks = gdf_sinks.copy()
            sorted_sinks['distance'] = sorted_sinks.distance(sources_in_cluster.unary_union)
            sorted_sinks = sorted_sinks.sort_values(by='distance')

            remaining_captured = total_captured

            for _, sink_row in sorted_sinks.iterrows():
                sink_name = sink_row['CO2 sinks']
                sink_capacity = sink_row['Cap']
                
                
                if sink_name in partial_sink_data:
                    current_total_captured = partial_sink_data[sink_name]['Total Captured']
                else:
                    current_total_captured = 0

                if remaining_captured + current_total_captured <= sink_capacity:
                    nearest_sink = sink_row.geometry
                    distances_to_sink = sources_in_cluster.distance(nearest_sink)
                    nearest_source = sources_in_cluster.loc[distances_to_sink.idxmin()].geometry
                    line = LineString([nearest_sink, nearest_source])
                    folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]),
                                            (line.coords[1][1], line.coords[1][0])], color="blue", weight=1).add_to(m2)
                    route_number = routes_with_capacity(remaining_captured, line.length*100)
                    segment_distance = route_number*line.length
                    distance_partial += segment_distance
                    
                    diameter, sink_pbf_power_partial, pressure_out = calculate_pipeline_diameter(total_captured*3.171e-5, pressure_in, line.length*100000, 285.15)
                    partial_sink_pipeline_cost = calculate_pipeline_cost(diameter, line.length*100000)              
                    partial_sink_pbf_cost = (8.35 * sink_pbf_power_partial + 0.49) * 1.67       
                    pressure_in = pressure_out
                    if pressure_in < 7.38e6:
                        pressure_in = 15e6

                    
                    EI_pbf_energy_sink = 882 * sink_pbf_power_partial / 3600 / 1000
                    EI_pipeline_fugitive_sink = 1.4 * line.length*100
                    EI_of_transportation_sink += (EI_pbf_energy_sink + EI_pipeline_fugitive_sink)*route_number
            
                   
                    EI_of_transp = EI_of_transportation_sink + EI_of_transportation_source
                    if sink_name in partial_sink_data:
                        partial_sink_data[sink_name]['Total Captured'] += remaining_captured
                        partial_sink_data[sink_name]['Total Emitted'] += total_emitted
                        partial_sink_data[sink_name]['Total Capture Cost'] += total_capture_cost
                        partial_sink_data[sink_name]['Total Distance'] += distance_partial
                        partial_sink_data[sink_name]['Storage Cost'] =0
                        partial_sink_data[sink_name]['n_wells'] = 0
                        partial_sink_data[sink_name]['EI_of_transp'] += EI_of_transp
                        partial_sink_data[sink_name]['EI_of_transp_inject'] += EI_of_transp
                        partial_sink_data[sink_name]['route_number'] += route_number
                        partial_sink_data[sink_name]['EI_inject'] = 0
                        partial_sink_data[sink_name]['Number of Sources'] += len(sources_in_cluster)
                        partial_sink_data[sink_name]['Total Pipeline Cost'] += partial_hub_pipeline_cost*route_number
                        partial_sink_data[sink_name]['Total Pipeline Cost'] += partial_sink_pipeline_cost
                        partial_sink_data[sink_name]['Total pipeline Booster Cost'] += partial_hub_pbf_cost*route_number
                        partial_sink_data[sink_name]['Total pipeline Booster Cost'] += partial_sink_pbf_cost
                    else:
                        partial_sink_data[sink_name] = {
                            'Total Captured': remaining_captured,
                            'Total Emitted': total_emitted,
                            'Total Capture Cost': total_capture_cost,
                            'Capacity': sink_capacity,
                            'Total Distance': distance_partial,
                            'Sink Lifetime': sink_capacity/remaining_captured,
                            'Storage Cost': 0,
                            'n_wells': 0,
                            'EI_of_transp': EI_of_transp,
                            'EI_of_transp_inject': EI_of_transp,
                            'route_number': route_number,
                            'EI_inject': 0,
                            'Number of Sources': len(sources_in_cluster),
                            'Total Pipeline Cost': partial_hub_pipeline_cost*route_number+partial_sink_pipeline_cost,
                            'Total pipeline Booster Cost': partial_hub_pbf_cost*route_number+partial_sink_pbf_cost
                        }
                    remaining_captured = 0
                    break
                else:
                    capture_for_this_sink = sink_capacity - current_total_captured
                    
                    nearest_sink = sink_row.geometry
                    distances_to_sink = sources_in_cluster.distance(nearest_sink)
                    nearest_source = sources_in_cluster.loc[distances_to_sink.idxmin()].geometry
                    line = LineString([nearest_sink, nearest_source])
                    folium.PolyLine(locations=[(line.coords[0][1], line.coords[0][0]),
                                            (line.coords[1][1], line.coords[1][0])], color="blue", dash_array='5, 4', weight=1).add_to(m2)
                    route_number = routes_with_capacity(remaining_captured, line.length*100)
                    segment_distance = route_number*line.length
                    distance_partial += segment_distance
                    
                    
                    diameter, sink_pbf_power_partial, pressure_out = calculate_pipeline_diameter(total_captured*3.171e-5, pressure_in, line.length*100000, 285.15)
                    partial_sink_pipeline_cost = calculate_pipeline_cost(diameter, line.length*100000)              
                    partial_sink_pbf_cost = (8.35 * sink_pbf_power_partial + 0.49) * 1.67       
                    pressure_in = pressure_out
                    if pressure_in < 7.38e6:
                        pressure_in = 15e6

                    
                    EI_pbf_energy_sink = 882 * sink_pbf_power_partial / 3600 / 1000
                    EI_pipeline_fugitive_sink = 1.4 * line.length*100
                    EI_of_transportation_sink += (EI_pbf_energy_sink + EI_pipeline_fugitive_sink)*route_number
            

                    EI_of_transp = EI_of_transportation_sink + EI_of_transportation_source
                    


                    if sink_name in partial_sink_data:
                        partial_sink_data[sink_name]['Total Captured'] += capture_for_this_sink
                        partial_sink_data[sink_name]['Total Emitted'] += total_emitted 
                        partial_sink_data[sink_name]['Total Capture Cost'] += total_capture_cost 
                        partial_sink_data[sink_name]['Total Distance'] += distance_partial
                        partial_sink_data[sink_name]['Storage Cost'] = 0
                        partial_sink_data[sink_name]['n_wells'] = 0
                        partial_sink_data[sink_name]['EI_of_transp'] +=EI_of_transp
                        partial_sink_data[sink_name]['EI_of_transp_inject'] +=EI_of_transp
                        partial_sink_data[sink_name]['route_number'] += route_number
                        partial_sink_data[sink_name]['EI_inject'] = 0
                        partial_sink_data[sink_name]['Number of Sources'] += len(sources_in_cluster)
                        partial_sink_data[sink_name]['Total Pipeline Cost'] += partial_hub_pipeline_cost*route_number
                        partial_sink_data[sink_name]['Total Pipeline Cost'] += partial_sink_pipeline_cost
                        partial_sink_data[sink_name]['Total pipeline Booster Cost'] += partial_hub_pbf_cost*route_number
                        partial_sink_data[sink_name]['Total pipeline Booster Cost'] += partial_sink_pbf_cost
                        

                    else:
                        partial_sink_data[sink_name] = {
                            'Total Captured': capture_for_this_sink,
                            'Total Emitted': total_emitted * (capture_for_this_sink / total_captured),
                            'Total Capture Cost': total_capture_cost * (capture_for_this_sink / total_captured),
                            'Capacity': sink_capacity,
                            'Total Distance': distance_partial,
                            'Sink Lifetime': sink_capacity/capture_for_this_sink,
                            'Storage Cost': 0,
                            'n_wells': 0,
                            'EI_of_transp': EI_of_transp,
                            'EI_of_transp_inject': EI_of_transp,
                            'route_number': route_number,
                            'EI_inject': 0,
                            'Number of Sources': len(sources_in_cluster),
                            'Total Pipeline Cost': partial_hub_pipeline_cost*route_number+sink_pipeline_cost,
                            'Total pipeline Booster Cost': partial_hub_pbf_cost*route_number+sink_pbf_cost 
                        }
                    
                    remaining_captured -= capture_for_this_sink
        total_distance_partial += distance_partial
    
    for sink_name, data in partial_sink_data.items():
        # Convert total captured to million tonnes for storage cost calculation
        total_captured_million = data['Total Captured'] / 1e6
        
        # Get sink's characteristics
        sink_row = gdf_sinks[gdf_sinks['CO2 sinks'] == sink_name].iloc[0]
        d_m = sink_row['d_m']
        ro_kgm3 = sink_row['ro kg/m3']
        sink_type = sink_row['Type']

        # Calculate the storage cost using the cumulative captured emissions
        storage_cost, n_wells, EI_inject_compression_energy = calculate_storage_cost(
            {'Type': sink_type, 'd_m': d_m, 'ro_kgm3': ro_kgm3}, total_captured_million
        )

        # Update the sink data dictionary
        data['Storage Cost'] = storage_cost
        data['n_wells'] = n_wells
        data['EI_of_transp_inject'] += EI_inject_compression_energy
        data['EI_inject']= EI_inject_compression_energy

    total_total_distance_partial = sum(data['Total Distance'] for data in partial_sink_data.values())
    total_captured_tCO2_partial = sum(data['Total Captured'] for data in partial_sink_data.values())
    total_emitted_tCO2_partial = sum(data['Total Emitted'] for data in partial_sink_data.values())
    total_capture_cost_MUSD_partial = sum(data['Total Capture Cost'] for data in partial_sink_data.values())
    partial_transportation_pipeline_cost_MUSD = sum(data['Total Pipeline Cost'] for data in partial_sink_data.values())
    partial_transportation_booster_MUSD = sum(data['Total pipeline Booster Cost'] for data in partial_sink_data.values())
    environmental_impact_m_partial = sum(data['EI_of_transp_inject'] for data in partial_sink_data.values())
    total_storage_cost_MUSD_partial = sum(data['Storage Cost'] for data in partial_sink_data.values())
    EI_of_transp = sum(data['EI_of_transp'] for data in partial_sink_data.values())
   
    EI_inject = sum(data['EI_inject'] for data in partial_sink_data.values())

    # Append the results to the plot_data dictionary
    plot_data2['eps'].append(eps)
    plot_data2['Total Captured tCO2'].append(total_captured_tCO2_partial)
    plot_data2['Total Emitted tCO2'].append(total_emitted_tCO2_partial)
    plot_data2['Total Distance'].append(total_total_distance_partial * 100)
    plot_data2['Environmental impact'].append(environmental_impact_m_partial)
    plot_data2['Transportation Cost MUSD'] = partial_transportation_pipeline_cost_MUSD + partial_transportation_booster_MUSD
    plot_data2['Storage Cost MUSD'].append(total_storage_cost_MUSD_partial)
    plot_data2['Total Capture Cost MUSD'].append(total_capture_cost_MUSD_partial)
    plot_data2['Total Cost MUSD'].append(total_capture_cost_MUSD_partial + partial_transportation_pipeline_cost_MUSD + partial_transportation_booster_MUSD + total_storage_cost_MUSD_partial)
    plot_data2['EI_of_transp'].append(EI_of_transp)
    plot_data2['EI_inject'].append(EI_inject)
    
    
   
    plot_df_partial = pd.DataFrame(plot_data2)
    
    st.write("### Partial Redirection Map")
    # Display the partial redirection map
    folium_static(m2)

    # Prepare partial_sink_data for display
    partial_sink_data_df = pd.DataFrame.from_dict(partial_sink_data, orient='index').reset_index()
    partial_sink_data_df.columns = ['Sink Name', 'Total Captured', 'Total Emitted', 'Total Capture Cost', 'Sink Capacity', 'Total Distance', 'Sink Lifetime', 'Storage Cost', 'n_wells', 'EI_of_transp','EI_of_transp_inject', 'route_number', 'EI_inject', 'Number of Sources', 'Total Pipeline Cost', 'Total pipeline Booster Cost']

    # Display the table
    st.table(partial_sink_data_df)
    st.write(plot_df_partial)

if __name__ == "__main__":
    main()
