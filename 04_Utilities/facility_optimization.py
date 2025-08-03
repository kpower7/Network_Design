import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import folium
from folium import plugins

# Load data
path = "Data/"
stores_df = pd.read_csv(path + 'stores.csv')
demand_df = pd.read_csv(path + 'demand.csv')
products_df = pd.read_csv(path + 'products.csv')
facilities_df = pd.read_csv(path + 'facilities.csv')

# Selected facilities
selected_facilities = ['FAC027', 'FAC047', 'FAC035']  # Phoenix, Dallas, Columbus
facilities_df = facilities_df[facilities_df['FacilityID'].isin(selected_facilities)]

# Calculate annual demand in pallets for each store
demand_df = demand_df.merge(products_df[['SKU', 'SKUs/pallet']], on='SKU', how='left')
demand_df['Demand_Pallets'] = demand_df['Demand'] / demand_df['SKUs/pallet']
store_demand = demand_df.groupby('StoreID')['Demand_Pallets'].sum().reset_index()

# Merge with stores data
store_demand = stores_df.merge(store_demand, on='StoreID', how='right')

# Define standard transport rates ($ per pallet per mile)
# Using a simple distance-based rate structure
base_rate = 2.0  # Base rate per pallet per mile

# Calculate distances and transport costs
transport_costs = {}
for _, facility in facilities_df.iterrows():
    facility_coords = (facility['Latitude'], facility['Longitude'])
    for _, store in store_demand.iterrows():
        store_coords = (store['Latitude'], store['Longitude'])
        # Calculate distance in miles (approximate)
        distance = np.sqrt(
            (facility['Latitude'] - store['Latitude'])**2 + 
            (facility['Longitude'] - store['Longitude'])**2
        ) * 69  # Convert to miles (approximate)
        
        # Calculate transport cost
        transport_costs[(facility['FacilityID'], store['StoreID'])] = distance * base_rate

# Create optimization model
model = gp.Model("Facility_Location")

# Sets
facilities = facilities_df['FacilityID'].tolist()
stores = store_demand['StoreID'].tolist()

# Decision Variables
x = model.addVars(transport_costs.keys(), vtype=GRB.CONTINUOUS, name="ship")
y = model.addVars(facilities, vtype=GRB.BINARY, name="open")

# Objective: Minimize total transportation cost
model.setObjective(
    gp.quicksum(transport_costs[i,j] * x[i,j] for i,j in transport_costs.keys()),
    GRB.MINIMIZE
)

# Constraints
# 1. Meet store demand
for s in stores:
    store_pallets = store_demand[store_demand['StoreID'] == s]['Demand_Pallets'].iloc[0]
    model.addConstr(
        gp.quicksum(x[f,s] for f in facilities if (f,s) in transport_costs) == store_pallets,
        f"demand_{s}"
    )

# 2. Facility capacity constraints
for f in facilities:
    capacity = facilities_df[facilities_df['FacilityID'] == f]['Size_SqFt'].iloc[0] / 25  # Approx pallets per sqft
    model.addConstr(
        gp.quicksum(x[f,s] for s in stores if (f,s) in transport_costs) <= capacity * y[f],
        f"capacity_{f}"
    )

# 3. Must use all three facilities
model.addConstr(gp.quicksum(y[f] for f in facilities) == 3, "use_all_facilities")

# Solve the model
model.optimize()

# Process results
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found!")
    
    # Print facility decisions
    print("\nFacility Decisions:")
    for f in facilities:
        if y[f].x > 0.5:
            total_volume = sum(x[f,s].x for s in stores if (f,s) in transport_costs)
            print(f"{f}: Open, Handling {total_volume:.0f} pallets")
    
    # Calculate total cost
    total_cost = model.objVal
    print(f"\nTotal Transportation Cost: ${total_cost:,.2f}")
    
    # Create visualization
    def create_network_map():
        # Create base map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Add facilities
        for _, facility in facilities_df.iterrows():
            if y[facility['FacilityID']].x > 0.5:
                folium.CircleMarker(
                    location=[facility['Latitude'], facility['Longitude']],
                    radius=10,
                    color='red',
                    fill=True,
                    popup=f"{facility['Location']} (Facility)",
                ).add_to(m)
        
        # Add stores and connections
        for _, store in store_demand.iterrows():
            folium.CircleMarker(
                location=[store['Latitude'], store['Longitude']],
                radius=5,
                color='blue',
                fill=True,
                popup=f"Store {store['StoreID']}"
            ).add_to(m)
            
            # Draw lines for active shipments
            for f in facilities:
                if (f, store['StoreID']) in transport_costs and x[f, store['StoreID']].x > 0:
                    facility_row = facilities_df[facilities_df['FacilityID'] == f].iloc[0]
                    folium.PolyLine(
                        locations=[
                            [facility_row['Latitude'], facility_row['Longitude']],
                            [store['Latitude'], store['Longitude']]
                        ],
                        color='gray',
                        weight=1,
                        opacity=0.5
                    ).add_to(m)
        
        return m
    
    # Create and save the map
    network_map = create_network_map()
    network_map.save('network_map.html')
    print("\nNetwork map has been saved as 'network_map.html'")
else:
    print("No optimal solution found")
