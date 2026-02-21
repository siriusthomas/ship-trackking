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
# Land/Water Check — uses global_land_mask (1km resolution raster)
# globe.is_ocean(lat, lon) returns True if the point is ocean/water
# =========================================================

from global_land_mask import globe as _globe

def is_on_land(lat, lon, _unused=None):
    """True if the coordinate is on land (1km resolution global raster)"""
    try:
        return not bool(_globe.is_ocean(lat, lon))
    except Exception:
        return False  # Default to water if check fails

def is_on_water(lat, lon):
    return not is_on_land(lat, lon)

def nudge_off_land(lat, lon, max_tries=16):
    """Spiral outward from a land point until water is found"""
    if not is_on_land(lat, lon):
        return lat, lon
    # Try increasing offsets in 8 compass directions
    for step in range(1, max_tries + 1):
        offset = step * 0.5
        for dlat, dlon in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nlat = lat + dlat * offset
            nlon = lon + dlon * offset
            if not is_on_land(nlat, nlon):
                return nlat, nlon
    return lat, lon  # Give up, keep original


# =========================================================
# Ship Generation Functions
# =========================================================

# Major maritime regions around the world
_MARITIME_REGIONS = [
    {'name': 'US East Coast',      'lat': 40.5,  'lon': -73.0,  'ships': 60, 'radius': 6},
    {'name': 'English Channel',    'lat': 50.5,  'lon':   0.5,  'ships': 55, 'radius': 4},
    {'name': 'Mediterranean',      'lat': 36.0,  'lon':  18.0,  'ships': 50, 'radius': 7},
    {'name': 'Persian Gulf',       'lat': 26.5,  'lon':  52.0,  'ships': 45, 'radius': 4},
    {'name': 'South China Sea',    'lat': 14.0,  'lon': 114.0,  'ships': 55, 'radius': 6},
    {'name': 'Indian Ocean',       'lat':  5.0,  'lon':  65.0,  'ships': 40, 'radius': 8},
    {'name': 'Singapore Strait',   'lat':  1.3,  'lon': 104.0,  'ships': 45, 'radius': 3},
    {'name': 'Gulf of Mexico',     'lat': 25.0,  'lon': -90.0,  'ships': 35, 'radius': 5},
    {'name': 'North Sea',          'lat': 56.0,  'lon':   3.0,  'ships': 35, 'radius': 4},
    {'name': 'East Africa Coast',  'lat': -3.0,  'lon':  42.0,  'ships': 30, 'radius': 5},
    {'name': 'Caribbean',          'lat': 16.0,  'lon': -65.0,  'ships': 30, 'radius': 5},
    {'name': 'Japan Sea',          'lat': 34.5,  'lon': 135.0,  'ships': 35, 'radius': 4},
    {'name': 'West Africa',        'lat':  3.0,  'lon':   3.0,  'ships': 25, 'radius': 5},
    {'name': 'Bay of Bengal',      'lat': 13.0,  'lon':  85.0,  'ships': 25, 'radius': 5},
    {'name': 'Red Sea',            'lat': 20.0,  'lon':  38.0,  'ships': 20, 'radius': 3},
    {'name': 'Australia East',     'lat': -30.0, 'lon': 153.0,  'ships': 20, 'radius': 4},
]

def generate_global_fleet():
    """Generate ships across multiple major maritime regions worldwide"""
    all_ships = []
    ship_types = ['Container', 'Tanker', 'Cargo', 'Passenger', 'Fishing']
    major_ports = [
        'New York', 'Rotterdam', 'Singapore', 'Shanghai', 'Dubai',
        'Tokyo', 'Mumbai', 'Cape Town', 'Sydney', 'Hamburg',
        'Piraeus', 'Hong Kong', 'Busan', 'Los Angeles', 'Santos'
    ]
    ship_id = 1

    for region in _MARITIME_REGIONS:
        for _ in range(region['ships']):
            # Generate random position within region's radius IN DEGREES
            # radius=5 ≈ 550km spread — visible on a global map
            for _attempt in range(100):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0.3, region['radius'])
                lat = region['lat'] + r * np.cos(angle)
                lon = region['lon'] + r * np.sin(angle) / max(0.1, np.cos(np.radians(region['lat'])))
                lat = max(-85, min(85, lat))
                if not is_on_land(lat, lon):
                    break

            ship_type = np.random.choice(ship_types)
            all_ships.append({
                'id': f'SHIP_{ship_id:03d}',
                'name': f'MV {ship_type} {ship_id}',
                'type': ship_type,
                'lat': lat,
                'lon': lon,
                'speed': np.random.uniform(5, 25),
                'heading': np.random.uniform(0, 360),
                'status': np.random.choice(['Active', 'Anchored', 'Underway']),
                'destination': np.random.choice(major_ports),
                'region': region['name'],
                'size': np.random.uniform(50, 400)
            })
            ship_id += 1

    return pd.DataFrame(all_ships)

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

    fig.update_layout(
        title={'text': '🚢 Maritime Dashboard - Click to Select Route Points', 'x': 0.5},
        mapbox=dict(style='open-street-map', center=dict(lat=20, lon=30), zoom=2),
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

ships_df = generate_global_fleet()
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
    
    # Keep zoom at global level — don't lock user into route zoom
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=2
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
    
    # Route B: Detour - scale detour with route distance
    dist = haversine_distance(start['lat'], start['lon'], end['lat'], end['lon'])
    detour = max(0.15, dist / 5000)  # bigger detour for longer routes
    route_b = generate_detour_route(start, end, detour, 20)
    risk_b = calculate_route_risk(route_b, risk_zones)
    routes.append({
        'waypoints': route_b,
        'distance': calculate_route_distance(route_b),
        'risk_score': risk_b,
        'safety_rating': get_safety_rating(risk_b)
    })
    
    route_c = generate_detour_route(start, end, -detour, 20)  # detour south
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
    """Optimal A* ocean path — shortest water route"""
    path = maritime_route(start, end)
    return smooth_path(path, num_points)

def generate_detour_route(start, end, detour_factor, num_points):
    """Route via ocean grid A* with latitude offset for genuine path variation.
    Larger multiplier (8×) ensures routes B/C diverge enough on the 3° grid
    to produce different ship densities and different risk scores."""
    # Clamp offset so destination stays on valid ocean grid
    mid_lat = (start['lat'] + end['lat']) / 2
    raw_offset = detour_factor * 8.0
    # Don't push destination into polar regions
    clamped_offset = max(-60 - mid_lat, min(60 - mid_lat, raw_offset))
    path = maritime_route(start, end, lat_offset=clamped_offset)
    return smooth_path(path, num_points)



def smooth_path(waypoints, num_points):
    """Interpolate between waypoints, nudging any land-hitting points to water."""
    if len(waypoints) < 2:
        return waypoints
    total = calculate_route_distance(waypoints)
    if total == 0:
        return waypoints
    result = []
    for i in range(len(waypoints) - 1):
        seg_dist = haversine_distance(
            waypoints[i]['lat'], waypoints[i]['lon'],
            waypoints[i+1]['lat'], waypoints[i+1]['lon']
        )
        seg_pts = max(2, int(num_points * seg_dist / total))
        for j in range(seg_pts):
            t = j / seg_pts
            lat = waypoints[i]['lat'] + t * (waypoints[i+1]['lat'] - waypoints[i]['lat'])
            lon = waypoints[i]['lon'] + t * (waypoints[i+1]['lon'] - waypoints[i]['lon'])
            # Nudge any interpolated point that falls on land back to water
            if is_on_land(lat, lon):
                lat, lon = nudge_off_land(lat, lon)
            result.append({'lat': lat, 'lon': lon})
    last = waypoints[-1]
    if is_on_land(last['lat'], last['lon']):
        nlat, nlon = nudge_off_land(last['lat'], last['lon'])
        result.append({'lat': nlat, 'lon': nlon})
    else:
        result.append(last)
    return result

# ─── Ocean Grid + A* Router ──────────────────────────────────────────────────
# Pre-compute a global ocean grid at 3° resolution once at startup.
# A* can only travel through ocean cells → routes can NEVER cross land.

_GRID_STEP = 1.5  # degrees — fine enough for narrow straits (Malacca, etc.)

print("Building global ocean grid (1.5° resolution)…")
_OCEAN_GRID = set()
for _la in np.arange(-78, 79, _GRID_STEP):
    for _lo in np.arange(-180, 181, _GRID_STEP):
        _la_r = round(float(_la), 1)
        _lo_r = round(float(_lo), 1)
        if is_on_water(_la_r, _lo_r):
            _OCEAN_GRID.add((_la_r, _lo_r))
print(f"Ocean grid ready: {len(_OCEAN_GRID)} ocean cells")

# Pre-index grid as a sorted list for fast nearest-neighbour search
_OCEAN_LIST = list(_OCEAN_GRID)

def _nearest_ocean_cell(lat, lon):
    """Return the ocean grid cell closest to (lat, lon)"""
    best, best_d = None, float('inf')
    for (la, lo) in _OCEAN_LIST:
        d = (la - lat) ** 2 + (lo - lon) ** 2  # squared Euclidean (fast, sufficient)
        if d < best_d:
            best_d = d
            best = (la, lo)
    return best

def maritime_route(start, end, lat_offset=0.0):
    """
    A* on the pre-computed ocean grid.
    Every cell in _OCEAN_GRID is verified water — route cannot cross land.
    lat_offset shifts the destination cell north/south for route variation.
    """
    import heapq

    src = _nearest_ocean_cell(start['lat'], start['lon'])
    dst = _nearest_ocean_cell(end['lat'] + lat_offset, end['lon'])

    if src is None or dst is None or src == dst:
        return [start, end]

    step = _GRID_STEP
    # 8-directional neighbours
    DIRS = [(0, step), (0, -step), (step, 0), (-step, 0),
            (step, step), (step, -step), (-step, step), (-step, -step)]

    def h(a, b):
        return haversine_distance(a[0], a[1], b[0], b[1])

    open_set = [(h(src, dst), 0.0, src)]
    came_from = {}
    g_score = {src: 0.0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == dst:
            break
        if g > g_score.get(current, float('inf')) + 0.1:
            continue
        la, lo = current
        for dla, dlo in DIRS:
            nb = (round(la + dla, 1), round(lo + dlo, 1))
            if nb not in _OCEAN_GRID:
                continue
            cost = g + haversine_distance(la, lo, nb[0], nb[1])
            if cost < g_score.get(nb, float('inf')):
                g_score[nb] = cost
                came_from[nb] = current
                heapq.heappush(open_set, (cost + h(nb, dst), cost, nb))

    # Reconstruct path
    if dst not in came_from and dst != src:
        # dst unreachable — try without lat_offset
        if lat_offset != 0:
            return maritime_route(start, end, lat_offset=0)
        return [start, end]

    path = []
    cur = dst
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.append(src)
    path.reverse()

    waypoints = [start]
    for la, lo in path:
        waypoints.append({'lat': la, 'lon': lo})
    waypoints.append(end)
    return waypoints



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
    """
    Risk score 0-100 = f(ships near this specific route).
    For each ship, find its minimum distance to any waypoint on the route.
    Ships very close = high contribution, farther away = smaller contribution.
    Routes through busy shipping lanes score much higher than empty ocean routes.
    """
    if len(route) == 0:
        return 5

    CLOSE_KM  = 50   # within this → strong obstacle
    MEDIUM_KM = 150  # within this → moderate obstacle
    FAR_KM    = 300  # within this → minor awareness

    ship_lats   = ships_df['lat'].values
    ship_lons   = ships_df['lon'].values
    ship_speeds = ships_df['speed'].values

    weighted_total = 0.0

    for idx in range(len(ship_lats)):
        slat, slon, sspeed = ship_lats[idx], ship_lons[idx], ship_speeds[idx]
        speed_w = 0.8 + (sspeed / 25.0) * 0.6  # 0.8–1.4×

        # Find the closest waypoint to this ship
        min_d = float('inf')
        for wp in route:
            d = haversine_distance(wp['lat'], wp['lon'], slat, slon)
            if d < min_d:
                min_d = d
            if min_d < 10:   # early exit if very close
                break

        if min_d < CLOSE_KM:
            weighted_total += speed_w * (1.0 - min_d / CLOSE_KM) * 3.0
        elif min_d < MEDIUM_KM:
            weighted_total += speed_w * (1.0 - min_d / MEDIUM_KM) * 1.0
        elif min_d < FAR_KM:
            weighted_total += speed_w * (1.0 - min_d / FAR_KM) * 0.2

    # Scale: 0 ships nearby → 5, ~10 nearby ships → ~45, ~25 ships → ~95
    score = 5 + int(weighted_total * 4.0)
    return max(5, min(score, 95))

def get_safety_rating(risk_score):
    """Convert risk score to safety rating"""
    if risk_score < 25:
        return 'Safe'
    elif risk_score < 55:
        return 'Moderate'
    else:
        return 'High Risk'

def add_click_layer(fig, ships_df):
    """Add invisible clickable grid covering major ocean areas worldwide"""
    # Cover the main ocean areas with multiple grids
    ocean_grids = [
        {'lat': 40, 'lon': -60, 'lat_r': 15, 'lon_r': 20},   # Atlantic
        {'lat': 20, 'lon': 80, 'lat_r': 25, 'lon_r': 25},    # Indian Ocean
        {'lat': 15, 'lon': -160, 'lat_r': 20, 'lon_r': 30},   # Pacific
        {'lat': 50, 'lon': 10, 'lat_r': 10, 'lon_r': 15},     # Europe/Med
        {'lat': -20, 'lon': 40, 'lat_r': 20, 'lon_r': 20},    # S Indian
    ]
    
    grid_lats = []
    grid_lons = []
    pts = 15  # points per grid dimension
    
    for g in ocean_grids:
        for i in range(pts):
            for j in range(pts):
                lat = g['lat'] - g['lat_r'] + (2 * g['lat_r'] * i / (pts - 1))
                lon = g['lon'] - g['lon_r'] + (2 * g['lon_r'] * j / (pts - 1))
                grid_lats.append(lat)
                grid_lons.append(lon)
    
    fig.add_trace(go.Scattermapbox(
        lat=grid_lats,
        lon=grid_lons,
        mode='markers',
        marker=dict(size=80, color='rgba(0,0,0,0)', opacity=0),
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
    print(f"   Ships: {len(ships_df)} across {len(_MARITIME_REGIONS)} regions")
    print(f"   Risk Zones: {len(risk_zones)}")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)