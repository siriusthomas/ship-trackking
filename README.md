# 🚢 Maritime Traffic Dashboard

A Flask-based interactive dashboard for real-time maritime traffic monitoring, risk analysis, and route optimization.

## 🌟 Key Features

- **Synthetic Fleet Generation**: Simulates a diverse fleet of ships (Container, Tanker, Cargo, Passenger, Fishing) with realistic movement data.
- **Risk Calculation & Zone Identification**: Uses **DBSCAN Clustering** to identify potential collision risk zones based on ship density.
- **Multi-Point Route Planning**: Users can click on the map to select multiple points for a custom ship route.
- **Advanced Route Optimization**: Implements the **Nearest Neighbor (TSP)** algorithm to optimize the order of waypoints for the shortest path.
- **Alternative Routing**: Generates three distinct route options (Direct, Northern Detour, Southern Detour) with associated risk scores and safety ratings.
- **Dynamic Visualizations**: Leveraging **Plotly Mapbox** for high-fidelity, interactive map interfaces.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly (Plotly.js + Plotly Python)
- **Frontend**: HTML5, CSS3, JavaScript

## 📂 Project Structure

- `app.py`: Main Flask application containing logic for data generation, risk analysis, and routing.
- `templates/`: HTML templates for the dashboard and index pages.
- `requirements.txt`: Python dependencies.
- `ship_trajectory_prediction.ipynb`: Jupyter notebooks for underlying trajectory prediction research.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Dashboard**:
   Open [http://localhost:5000](http://localhost:5000) in your web browser.

## 🛡️ Risk Assessment Scale

- **Safe**: Risk score < 25
- **Moderate**: Risk score between 25 and 55
- **High Risk**: Risk score > 55
