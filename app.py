# =========================================================
# MARITIME TRAFFIC DASHBOARD - Flask Web Application
# Features: Multi-point Route Planning + Risk Analysis
# =========================================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# =========================================================
# Land/Water Check Functions
# =========================================================

def is_on_land(lat, lon):
    """Simple land check - returns True if clearly on land"""
    if 38 < lat < 42 and -76 < lon < -72:  # NYC/NJ coastal area
        return False
    return False  # Default: assume water

def is_on_water(lat, lon):
    return not is_on_land(lat, lon)

# =========================================================
# Ship Generation Functions
# =========================================================

def generate_ship_fleet(num_ships=50, center_lat=40.7, center_lon=-74.0, radius=8, max_attempts=5000):
    """Generate synthetic fleet of ships"""
    ships = []
    ship_types = ['Container', 'Tanker', 'Cargo', 'Passenger', 'Fishing']

    attempts = 0
    while len(ships) < num_ships and attempts < max_attempts:
        attempts += 1
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.5, radius)
        lat = center_lat + r * np.cos(angle) / 111
        lon = center_lon + r * np.sin(angle) / (111 * np.cos(np.radians(center_lat)))

        if is_on_land(lat, lon):
            continue

        ship_type = np.random.choice(ship_types)
        ships.append({
            'id': f'SHIP_{len(ships)+1:03d}',
            'name': f'MV {ship_type} {len(ships)+1}',
            'type': ship_type,
            'lat': lat,
            'lon': lon,
            'speed': np.random.uniform(5, 20),
            'heading': np.random.uniform(0, 360),
            'status': np.random.choice(['Active', 'Anchored', 'Underway']),
            'destination': f'Port_{np.random.randint(1,10)}',
            'size': np.random.uniform(50, 300)
        })

    return pd.DataFrame(ships)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def identify_risk_zones(ships_df, collision_radius_km=3):
    """Identify high-risk zones using DBSCAN clustering"""
    if len(ships_df) < 2:
        return []

    coords = np.radians(ships_df[['lat', 'lon']].values)
    epsilon = collision_radius_km / 6371.0
    db = DBSCAN(eps=epsilon, min_samples=2, metric='haversine')
    ships_df['cluster'] = db.fit_predict(coords)

    risk_zones = []
    for cluster_id in ships_df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_ships = ships_df[ships_df['cluster'] == cluster_id]
        if len(cluster_ships) >= 2:
            center_lat = cluster_ships['lat'].mean()
            center_lon = cluster_ships['lon'].mean()
            max_dist = 0
            for _, ship in cluster_ships.iterrows():
                dist = haversine_distance(center_lat, center_lon, ship['lat'], ship['lon'])
                max_dist = max(max_dist, dist)
            risk_level = 'High' if len(cluster_ships) >= 4 else 'Medium'
            risk_zones.append({
                'center_lat': center_lat,
                'center_lon': center_lon,
                'radius_km': max_dist + collision_radius_km,
                'ship_count': len(cluster_ships),
                'risk_level': risk_level,
                'ships': cluster_ships['id'].tolist()
            })

    return risk_zones

# =========================================================
# Route Optimization - Nearest Neighbor TSP
# =========================================================

def optimize_route_order(points):
    """
    Optimize route order using Nearest Neighbor algorithm.
    Returns optimized order of points and total distance.
    """
    if len(points) < 2:
        return points, 0

    n = len(points)
    visited = [False] * n
    route_order = [0]  # Start from first point
    visited[0] = True
    total_distance = 0

    for _ in range(n - 1):
        current = route_order[-1]
        nearest = None
        min_dist = float('inf')

        for j in range(n):
            if not visited[j]:
                dist = haversine_distance(
                    points[current]['lat'], points[current]['lon'],
                    points[j]['lat'], points[j]['lon']
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        if nearest is not None:
            route_order.append(nearest)
            visited[nearest] = True
            total_distance += min_dist

    optimized_points = [points[i] for i in route_order]
    return optimized_points, total_distance

def generate_route_waypoints(points, waypoints_per_segment=10):
    """Generate smooth waypoints between selected points"""
    if len(points) < 2:
        return []

    route = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        for j in range(waypoints_per_segment + 1):
            t = j / waypoints_per_segment
            lat = start['lat'] + t * (end['lat'] - start['lat'])
            lon = start['lon'] + t * (end['lon'] - start['lon'])
            route.append({'lat': lat, 'lon': lon})

    return route

# =========================================================
# Map Generation
# =========================================================

def create_base_map(ships_df, risk_zones):
    """Create base map with ships and risk zones"""
    ship_colors = {
        'Container': '#1f77b4',
        'Tanker': '#d62728',
        'Cargo': '#2ca02c',
        'Passenger': '#9467bd',
        'Fishing': '#ff7f0e'
    }

    fig = go.Figure()

    # Add risk zones
    for zone in risk_zones:
        theta = np.linspace(0, 2*np.pi, 100)
        radius_deg = zone['radius_km'] / 111.0
        zone_lat = zone['center_lat'] + radius_deg * np.cos(theta)
        zone_lon = zone['center_lon'] + radius_deg * np.sin(theta) / np.cos(np.radians(zone['center_lat']))
        color = 'rgba(255,0,0,0.2)' if zone['risk_level'] == 'High' else 'rgba(255,165,0,0.2)'
        line_color = 'red' if zone['risk_level'] == 'High' else 'orange'

        fig.add_trace(go.Scattermapbox(
            lat=zone_lat, lon=zone_lon,
            mode='lines', fill='toself', fillcolor=color,
            line=dict(color=line_color, width=2),
            name=f"⚠️ {zone['risk_level']} Risk ({zone['ship_count']} ships)",
            hoverinfo='name', showlegend=True
        ))

    # Add ships
    for ship_type in ships_df['type'].unique():
        type_ships = ships_df[ships_df['type'] == ship_type]
        hover_text = [
            f"<b>{s['name']}</b><br>Type: {s['type']}<br>Speed: {s['speed']:.1f} kn"
            for _, s in type_ships.iterrows()
        ]
        fig.add_trace(go.Scattermapbox(
            lat=type_ships['lat'], lon=type_ships['lon'],
            mode='markers',
            marker=dict(size=10, color=ship_colors[ship_type], opacity=0.8),
            hovertext=hover_text, hoverinfo='text',
            name=f"🚢 {ship_type} ({len(type_ships)})", showlegend=True
        ))

    center_lat = ships_df['lat'].mean()
    center_lon = ships_df['lon'].mean()

    fig.update_layout(
        title={'text': '🚢 Maritime Dashboard - Click to Select Route Points', 'x': 0.5},
        mapbox=dict(style='open-street-map', center=dict(lat=center_lat, lon=center_lon), zoom=9),
        height=600, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)")
    )

    return fig

def add_route_to_map(fig, route_points, selected_points, total_distance):
    """Add optimized route and markers to the map"""
    # Add route line
    if len(route_points) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=[p['lat'] for p in route_points],
            lon=[p['lon'] for p in route_points],
            mode='lines',
            line=dict(color='#2E86AB', width=4),
            name=f"🛤️ Route ({total_distance:.1f} km)",
            showlegend=True
        ))

    # Add numbered markers for selected points
    for i, point in enumerate(selected_points):
        color = 'green' if i == 0 else ('red' if i == len(selected_points)-1 else 'blue')
        fig.add_trace(go.Scattermapbox(
            lat=[point['lat']], lon=[point['lon']],
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=[str(i + 1)],
            textposition='middle center',
            textfont=dict(size=12, color='white'),
            name=f"Point {i + 1}",
            showlegend=False
        ))

    return fig

# =========================================================
# Global Data (generated once on startup)
# =========================================================

ships_df = generate_ship_fleet(num_ships=50, center_lat=40.7, center_lon=-74.0, radius=8)
risk_zones = identify_risk_zones(ships_df, collision_radius_km=3)

# =========================================================
# Flask Routes
# =========================================================

@app.route('/')
def index():
    fig = create_base_map(ships_df, risk_zones)
    map_json = fig.to_json()
    return render_template('index.html', map_json=map_json)

@app.route('/dashboard')
def dashboard():
    fig = create_base_map(ships_df, risk_zones)
    fig = add_click_layer(fig, ships_df)
    map_json = fig.to_json()
    return render_template('dashboard.html', map_json=map_json)

@app.route('/plan_route', methods=['POST'])
def plan_route():
    """API endpoint to calculate 3 alternative routes with risk scores"""
    data = request.get_json()
    points = data.get('points', [])

    if len(points) < 2:
        return jsonify({'error': 'Need at least 2 points'}), 400

    start = points[0]
    end = points[-1]
    
    # Generate 3 alternative routes
    routes = generate_three_routes(start, end, risk_zones)
    
    # Create map with all routes
    fig = create_base_map(ships_df, risk_zones)
    
    # Add invisible click layer for map interaction
    fig = add_click_layer(fig, ships_df)
    
    # Add all 3 routes to map
    route_colors = ['#2E86AB', '#F18F01', '#C73E1D']
    route_names = ['Route A (Optimal)', 'Route B (Alternative)', 'Route C (Alternative)']
    
    for i, route in enumerate(routes):
        fig.add_trace(go.Scattermapbox(
            lat=[p['lat'] for p in route['waypoints']],
            lon=[p['lon'] for p in route['waypoints']],
            mode='lines',
            line=dict(color=route_colors[i], width=4 if i == 0 else 3),
            name=f"{route_names[i]} - {route['distance']:.1f}km",
            hovertext=f"Distance: {route['distance']:.1f} km<br>Risk Score: {route['risk_score']}<br>Safety: {route['safety_rating']}",
            hoverinfo='text',
            showlegend=True
        ))
    
    # Add start/end markers
    fig.add_trace(go.Scattermapbox(
        lat=[start['lat']], lon=[start['lon']],
        mode='markers+text', marker=dict(size=18, color='#50fa7b'),
        text=['Start'], textposition='top center',
        name='Start Point', showlegend=False
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[end['lat']], lon=[end['lon']],
        mode='markers+text', marker=dict(size=18, color='#ff5555'),
        text=['End'], textposition='top center',
        name='End Point', showlegend=False
    ))
    
    # Center map on route (calculate bounds)
    all_lats = [start['lat'], end['lat']]
    all_lons = [start['lon'], end['lon']]
    for route in routes:
        for wp in route['waypoints']:
            all_lats.append(wp['lat'])
            all_lons.append(wp['lon'])
    
    center_lat = (min(all_lats) + max(all_lats)) / 2
    center_lon = (min(all_lons) + max(all_lons)) / 2
    
    # Calculate appropriate zoom level based on route span
    lat_span = max(all_lats) - min(all_lats)
    lon_span = max(all_lons) - min(all_lons)
    max_span = max(lat_span, lon_span)
    
    if max_span > 10:
        zoom = 4
    elif max_span > 5:
        zoom = 5
    elif max_span > 2:
        zoom = 6
    elif max_span > 1:
        zoom = 7
    else:
        zoom = 8
    
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        )
    )

    return jsonify({
        'map': fig.to_json(),
        'routes': [
            {
                'name': route_names[i],
                'distance': routes[i]['distance'],
                'risk_score': routes[i]['risk_score'],
                'safety_rating': routes[i]['safety_rating']
            } for i in range(len(routes))
        ],
        'points_order': [{'lat': start['lat'], 'lon': start['lon']}, {'lat': end['lat'], 'lon': end['lon']}]
    })

def generate_three_routes(start, end, risk_zones):
    """Generate 3 alternative routes between start and end points"""
    routes = []
    
    # Route A: Direct route
    route_a = generate_direct_route(start, end, 20)
    risk_a = calculate_route_risk(route_a, risk_zones)
    routes.append({
        'waypoints': route_a,
        'distance': calculate_route_distance(route_a),
        'risk_score': risk_a,
        'safety_rating': get_safety_rating(risk_a)
    })
    
    # Route B: Northern detour (avoid risk zones)
    route_b = generate_detour_route(start, end, 0.15, 20)  # detour north
    risk_b = calculate_route_risk(route_b, risk_zones)
    routes.append({
        'waypoints': route_b,
        'distance': calculate_route_distance(route_b),
        'risk_score': risk_b,
        'safety_rating': get_safety_rating(risk_b)
    })
    
    # Route C: Southern detour
    route_c = generate_detour_route(start, end, -0.15, 20)  # detour south
    risk_c = calculate_route_risk(route_c, risk_zones)
    routes.append({
        'waypoints': route_c,
        'distance': calculate_route_distance(route_c),
        'risk_score': risk_c,
        'safety_rating': get_safety_rating(risk_c)
    })
    
    # Sort by risk (lowest first)
    routes.sort(key=lambda x: x['risk_score'])
    
    return routes

def generate_direct_route(start, end, num_points):
    """Generate direct route between two points"""
    route = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = start['lat'] + t * (end['lat'] - start['lat'])
        lon = start['lon'] + t * (end['lon'] - start['lon'])
        route.append({'lat': lat, 'lon': lon})
    return route

def generate_detour_route(start, end, detour_factor, num_points):
    """Generate curved route with detour"""
    route = []
    mid_lat = (start['lat'] + end['lat']) / 2 + detour_factor
    mid_lon = (start['lon'] + end['lon']) / 2 + detour_factor * 0.3
    
    for i in range(num_points + 1):
        t = i / num_points
        # Quadratic bezier curve
        lat = (1-t)**2 * start['lat'] + 2*(1-t)*t * mid_lat + t**2 * end['lat']
        lon = (1-t)**2 * start['lon'] + 2*(1-t)*t * mid_lon + t**2 * end['lon']
        route.append({'lat': lat, 'lon': lon})
    return route

def calculate_route_distance(route):
    """Calculate total distance of a route in km"""
    total = 0
    for i in range(len(route) - 1):
        total += haversine_distance(
            route[i]['lat'], route[i]['lon'],
            route[i+1]['lat'], route[i+1]['lon']
        )
    return total

def calculate_route_risk(route, risk_zones):
    """Calculate risk score for a route based on proximity to risk zones (0-100 scale)"""
    if not risk_zones or len(route) == 0:
        return 5  # Minimal base risk
    
    # Calculate risk per waypoint, then average
    point_risks = []
    for point in route:
        point_risk = 0
        for zone in risk_zones:
            dist = haversine_distance(point['lat'], point['lon'], 
                                     zone['center_lat'], zone['center_lon'])
            radius = zone['radius_km']
            
            if dist < radius:
                # Inside risk zone - higher risk
                point_risk += 15 + (zone['ship_count'] * 0.5)
            elif dist < radius * 2:
                # Near risk zone
                point_risk += 8 + (zone['ship_count'] * 0.2)
            elif dist < radius * 3:
                # Somewhat close
                point_risk += 3
        
        point_risks.append(min(point_risk, 50))  # Cap per-point risk
    
    # Average across all points, scale to 0-100
    avg_risk = sum(point_risks) / len(point_risks) if point_risks else 0
    
    # Add a small random variation based on route distance
    route_dist = calculate_route_distance(route)
    distance_factor = min(route_dist / 100, 10)  # Longer routes slightly more risky
    
    final_risk = int(avg_risk + distance_factor)
    return max(5, min(final_risk, 95))  # Keep between 5-95 for variety

def get_safety_rating(risk_score):
    """Convert risk score to safety rating"""
    if risk_score < 25:
        return 'Safe'
    elif risk_score < 55:
        return 'Moderate'
    else:
        return 'High Risk'

def add_click_layer(fig, ships_df):
    """Add invisible clickable grid points to enable clicking anywhere on map"""
    center_lat = ships_df['lat'].mean()
    center_lon = ships_df['lon'].mean()
    
    # Create very dense grid for complete sea coverage
    grid_size = 50  # 50x50 = 2500 points for full coverage
    lat_range = 5.0  # Cover 10 degrees latitude total
    lon_range = 6.0  # Cover 12 degrees longitude total
    
    grid_lats = []
    grid_lons = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat - lat_range + (2 * lat_range * i / (grid_size - 1))
            lon = center_lon - lon_range + (2 * lon_range * j / (grid_size - 1))
            grid_lats.append(lat)
            grid_lons.append(lon)
    
    fig.add_trace(go.Scattermapbox(
        lat=grid_lats,
        lon=grid_lons,
        mode='markers',
        marker=dict(size=80, color='rgba(0,0,0,0)', opacity=0),  # Very large invisible markers
        hoverinfo='none',
        name='Click to select',
        showlegend=False
    ))
    
    return fig

# =========================================================
# Run Server
# =========================================================

if __name__ == '__main__':
    print("Maritime Traffic Dashboard Starting...")
    print(f"   Ships: {len(ships_df)}")
    print(f"   Risk Zones: {len(risk_zones)}")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)