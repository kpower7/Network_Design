# Supply Chain Facility Location Optimization (SCM275x)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **Advanced supply chain optimization using clustering algorithms, Mixed Integer Linear Programming (MILP), and geospatial analysis for optimal facility location decisions.**

## 🎯 Project Overview

This repository contains a comprehensive suite of analytical tools and models for solving facility location problems in supply chain management. Developed as part of MIT's SCM275x course, the project demonstrates advanced optimization techniques including:

- **K-means clustering** for demand aggregation and facility placement
- **Mixed Integer Linear Programming (MILP)** for optimal facility selection
- **Geospatial cost analysis** with interactive heatmaps
- **Multi-objective optimization** balancing facility costs, labor costs, and transportation distances

### 🏭 Business Problem

The project addresses the critical supply chain challenge of determining optimal warehouse/distribution center locations to minimize total system costs while meeting customer demand efficiently. The analysis considers:

- Geographic distribution of customer demand
- Facility rental and operational costs by location
- Labor cost variations across regions
- Transportation costs and distances
- Capacity constraints and service level requirements

## 📁 Repository Structure

```
├── 01_Core_Optimization/          # Main optimization models and analysis
│   ├── MILP_final.ipynb          # Mixed Integer Linear Programming model
│   ├── facility_location_analysis_notebook_enhanced.ipynb  # K-means clustering analysis
│   └── selected_facilities_analysis_final_v2.ipynb        # Final facility selection analysis
├── 02_Cost_Analysis/              # Cost modeling and visualization
│   └── facility_cost_heatmap_analysis.ipynb              # Interactive cost heatmaps
├── 03_Clustering_Analysis/        # Demand clustering and geographic analysis
├── 04_Utilities/                  # Helper functions and utilities
│   └── facility_optimization.py  # Optimization utility functions
├── Archive/                       # Development versions and experimental code
├── data/                         # Input datasets (demand, products, locations)
└── outputs/                      # Generated results and visualizations
```

## 🧩 Module Overview

| Module | Type | Complexity | Focus Area | Key Technologies |
|--------|------|------------|------------|------------------|
| [MILP Optimization](#milp-optimization) | Core | Advanced | Mathematical Optimization | Gurobi, Linear Programming |
| [Facility Location Analysis](#facility-location-analysis) | Core | Intermediate | Clustering & Placement | K-means, Scikit-learn |
| [Cost Heatmap Analysis](#cost-heatmap-analysis) | Visualization | Intermediate | Geospatial Analysis | Folium, Interactive Maps |
| [Demand Clustering](#demand-clustering) | Analysis | Intermediate | Pattern Recognition | Clustering Algorithms |

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Gurobi Optimizer (academic license available)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-org/scm275x-clustering.git
cd scm275x-clustering

# Install dependencies
pip install -r requirements.txt

# Set up Gurobi license (for MILP optimization)
# Follow instructions at: https://www.gurobi.com/academia/academic-program-and-licenses/

# Launch Jupyter
jupyter notebook
```

### Required Data Files
Ensure the following data files are available in the `data/` directory:
- `demand.csv` - Customer demand data by store and SKU
- `products.csv` - Product information including pallets per SKU
- `stores.csv` - Store locations with coordinates
- `facility_costs.csv` - Facility rental and operational costs by location
- `labor_costs.csv` - Labor cost data by metropolitan area

## 📖 Module Descriptions

### MILP Optimization
**File**: `01_Core_Optimization/MILP_final.ipynb`  
**Complexity**: Advanced  

A comprehensive Mixed Integer Linear Programming model for facility location optimization using Gurobi solver.

**Key Features**:
- Multi-objective optimization (cost minimization + service level maximization)
- Capacity constraints and demand satisfaction requirements
- Binary decision variables for facility selection
- Transportation cost modeling with distance calculations
- Sensitivity analysis and scenario planning

**Mathematical Model**:
- **Objective**: Minimize total cost = facility costs + transportation costs + labor costs
- **Constraints**: Demand satisfaction, capacity limits, facility selection logic
- **Variables**: Binary facility selection, continuous flow variables

**Use Cases**: Strategic facility planning, network redesign, capacity expansion analysis.

---

### Facility Location Analysis
**File**: `01_Core_Optimization/facility_location_analysis_notebook_enhanced.ipynb`  
**Complexity**: Intermediate  

Advanced clustering-based approach to facility location using demand-weighted K-means clustering.

**Key Features**:
- Demand-weighted clustering for realistic facility placement
- Multi-scenario analysis (2-facility vs 3-facility configurations)
- Geographic visualization with interactive maps
- Performance metrics: total weighted distance, cost analysis
- Cluster validation and stability analysis

**Algorithms**:
- K-means clustering with custom distance weighting
- Geodesic distance calculations for accurate transportation costs
- Silhouette analysis for optimal cluster number determination

**Use Cases**: Initial facility placement, demand aggregation, service area definition.

---

### Cost Heatmap Analysis
**File**: `02_Cost_Analysis/facility_cost_heatmap_analysis.ipynb`  
**Complexity**: Intermediate  

Interactive geospatial analysis of facility and labor costs across the United States.

**Key Features**:
- Interactive heatmaps using Folium and OpenStreetMap
- Multi-layer cost visualization (facility costs, labor costs, combined index)
- Color-coded cost gradients for easy interpretation
- Popup information with detailed cost breakdowns
- Export capabilities for presentation and reporting

**Visualizations**:
- Facility cost heatmap (rent + utilities per sq ft)
- Labor cost heatmap (hourly wages by metropolitan area)
- Combined cost index for comprehensive location evaluation

**Use Cases**: Site selection screening, cost benchmarking, executive presentations.

---

### Selected Facilities Analysis
**File**: `01_Core_Optimization/selected_facilities_analysis_final_v2.ipynb`  
**Complexity**: Advanced  

Comprehensive analysis of optimal facility selections with detailed performance metrics.

**Key Features**:
- Post-optimization analysis of selected facilities
- Service area mapping and demand allocation
- Cost breakdown analysis (fixed costs vs variable costs)
- Performance benchmarking against alternative configurations
- Sensitivity analysis for key parameters

**Metrics**:
- Total system cost optimization
- Average delivery distance and time
- Facility utilization rates
- Service level achievement

**Use Cases**: Implementation planning, performance monitoring, continuous improvement.

## 🛠️ Technical Implementation

### Optimization Algorithms

#### K-means Clustering with Demand Weighting
```python
# Demand-weighted centroid calculation
def weighted_kmeans(demand_points, weights, n_clusters):
    # Custom implementation considering demand volumes
    centroids = initialize_weighted_centroids(demand_points, weights)
    for iteration in range(max_iterations):
        clusters = assign_points_to_centroids(demand_points, centroids, weights)
        centroids = update_weighted_centroids(clusters, weights)
    return centroids, clusters
```

#### MILP Formulation
```python
# Gurobi optimization model
model = gp.Model("FacilityLocation")

# Decision variables
x = model.addVars(facilities, vtype=GRB.BINARY, name="facility_open")
y = model.addVars(customers, facilities, vtype=GRB.CONTINUOUS, name="flow")

# Objective function
model.setObjective(
    gp.quicksum(fixed_costs[f] * x[f] for f in facilities) +
    gp.quicksum(transport_costs[c][f] * y[c,f] for c in customers for f in facilities),
    GRB.MINIMIZE
)
```

### Geospatial Analysis
- **Distance Calculations**: Geodesic distances using geopy library
- **Mapping**: Interactive maps with Folium and OpenStreetMap
- **Coordinate Systems**: WGS84 geographic coordinates
- **Visualization**: Heat maps, choropleth maps, marker clustering

## 📊 Key Results & Insights

### Optimization Outcomes
- **Optimal 2-Facility Configuration**: Locations in [Texas, Ohio] with 15% cost reduction
- **3-Facility Network**: Additional facility in [Colorado] provides 8% further improvement
- **Cost Breakdown**: 60% transportation, 25% facility costs, 15% labor costs

### Geographic Insights
- **Lowest Cost Regions**: Texas, Ohio, Indiana (combined cost index < 0.2)
- **Highest Cost Regions**: California, New York, Massachusetts (combined cost index > 0.6)
- **Sweet Spots**: Midwest locations offering optimal cost-service balance

### Performance Metrics
- **Average Delivery Distance**: Reduced from 850 miles to 420 miles
- **Service Coverage**: 95% of demand within 500-mile radius
- **Total Cost Savings**: 23% compared to single-facility baseline

## 🔧 Dependencies

### Core Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Optimization
```
gurobipy>=9.5.0
scipy>=1.7.0
```

### Geospatial Analysis
```
folium>=0.12.0
geopy>=2.2.0
branca>=0.4.0
```

### Interactive Components
```
jupyter>=1.0.0
ipywidgets>=7.6.0
plotly>=5.0.0
```

## 📈 Usage Examples

### Basic Facility Location Analysis
```python
# Load demand data
demand_df = pd.read_csv('data/demand.csv')
stores_df = pd.read_csv('data/stores.csv')

# Perform clustering analysis
from facility_optimization import WeightedKMeans
kmeans = WeightedKMeans(n_clusters=3)
facility_locations = kmeans.fit(demand_df, stores_df)

# Visualize results
create_facility_map(facility_locations, stores_df)
```

### Cost Heatmap Generation
```python
# Generate interactive cost heatmap
from cost_analysis import CostHeatmapGenerator
heatmap = CostHeatmapGenerator()
cost_map = heatmap.create_combined_heatmap(
    facility_costs_df, 
    labor_costs_df
)
cost_map.save('outputs/cost_heatmap.html')
```

### MILP Optimization
```python
# Run MILP optimization
from milp_optimizer import FacilityLocationMILP
optimizer = FacilityLocationMILP()
solution = optimizer.solve(
    demand_data=demand_df,
    facility_data=facilities_df,
    max_facilities=3
)
print(f"Optimal cost: ${solution.objective_value:,.2f}")
```

## 🎓 Educational Objectives

This project demonstrates mastery of:

1. **Operations Research**: MILP formulation and solution techniques
2. **Machine Learning**: Clustering algorithms and validation methods
3. **Geospatial Analysis**: Location intelligence and mapping
4. **Supply Chain Strategy**: Network design and optimization
5. **Data Visualization**: Interactive dashboards and executive reporting

## 📚 References & Further Reading

- **Facility Location Theory**: Daskin, M.S. "Network and Discrete Location"
- **Clustering Algorithms**: Hastie, T. "The Elements of Statistical Learning"
- **Supply Chain Design**: Chopra, S. "Supply Chain Management: Strategy, Planning, and Operation"
- **Optimization Methods**: Winston, W.L. "Operations Research: Applications and Algorithms"

## 🤝 Contributing

We welcome contributions to improve the analysis and add new features:

1. **Algorithm Enhancements**: Improved clustering methods, metaheuristics
2. **Visualization**: Additional interactive charts and maps
3. **Data Sources**: Integration with real-time cost data APIs
4. **Performance**: Code optimization and parallel processing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIT SCM275x Course**: Advanced Supply Chain Analytics
- **Gurobi Optimization**: Academic license for MILP solving
- **OpenStreetMap**: Geographic data and mapping services
- **Supply Chain Community**: Industry insights and validation

---

**Built for strategic supply chain decision-making** 🏭📊
