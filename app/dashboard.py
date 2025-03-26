import dash
from dash import dcc, html, Input, Output
import dash_leaflet as dl
import plotly.express as px
import pandas as pd
import geopandas as gpd
import networkx as nx
import pickle
from datetime import datetime

# Load data
with open('data/processed/transport_graph.pkl', 'rb') as f:
    graph = pickle.load(f)

with open('data/processed/aco_results.pkl', 'rb') as f:
    aco_results = pickle.load(f)

bus_stops = pd.read_csv('data/raw/bus_stops.csv')
bus_routes = pd.read_csv('data/raw/bus_routes.csv')
fuel_data = pd.read_csv('data/raw/fuel_consumption.csv')

# Convert route stops from string to list
bus_routes['stops'] = bus_routes['stops'].apply(
    lambda x: x.strip("[]").replace("'", "").split(", "))

# Create Dash app
app = dash.Dash(__name__, title="Transport Optimization Dashboard")

# Define layout
app.layout = html.Div([
    html.H1("Urban Public Transport Optimization Dashboard", 
           style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='route-selector',
                options=[{'label': f"Route {i+1}", 'value': route_id} 
                        for i, route_id in enumerate(bus_routes['route_id'])],
                value=bus_routes['route_id'].iloc[0],
                style={'width': '100%'}
            ),
            dl.Map(
                id='map',
                center=[bus_stops['latitude'].mean(), bus_stops['longitude'].mean()],
                zoom=12,
                style={'height': '500px', 'width': '100%'},
                children=[
                    dl.TileLayer(),
                    dl.LayerGroup(id='route-layer'),
                    dl.LayerGroup(id='stops-layer')
                ]
            )
        ], style={'width': '60%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='fuel-efficiency-chart'),
            dcc.Graph(id='passenger-load-chart'),
            html.Div([
                html.H3("ACO Optimization Results"),
                html.Pre(id='aco-results')
            ])
        ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    
    html.Div([
        dcc.Graph(id='route-comparison-chart')
    ])
])

# Callbacks
@app.callback(
    [Output('route-layer', 'children'),
     Output('stops-layer', 'children')],
    [Input('route-selector', 'value')]
)
def update_map(selected_route):
    # Get selected route data
    route = bus_routes[bus_routes['route_id'] == selected_route].iloc[0]
    stops = route['stops']
    
    # Create route line
    route_line = []
    for i in range(len(stops)-1):
        start = bus_stops[bus_stops['stop_id'] == stops[i]].iloc[0]
        end = bus_stops[bus_stops['stop_id'] == stops[i+1]].iloc[0]
        
        route_line.append([
            [start['longitude'], start['latitude']],
            [end['longitude'], end['latitude']]
        ])
    
    # Create route layer
    route_layer = dl.Polyline(
        positions=route_line,
        color='blue',
        weight=3
    )
    
    # Create stops layer
    stops_layer = []
    for stop_id in stops:
        stop = bus_stops[bus_stops['stop_id'] == stop_id].iloc[0]
        stops_layer.append(
            dl.CircleMarker(
                center=[stop['latitude'], stop['longitude']],
                radius=8,
                color='red',
                fill=True,
                fillColor='red',
                children=dl.Tooltip(stop['stop_name'])
            )
        )
    
    # Highlight terminals
    terminals = [n for n, attr in graph.nodes(data=True) if attr['is_terminal']]
    for stop_id in terminals:
        if stop_id in stops:
            stop = bus_stops[bus_stops['stop_id'] == stop_id].iloc[0]
            stops_layer.append(
                dl.CircleMarker(
                    center=[stop['latitude'], stop['longitude']],
                    radius=12,
                    color='green',
                    fill=True,
                    fillColor='green',
                    children=dl.Tooltip(f"Terminal: {stop['stop_name']}")
                )
            )
    
    return [route_layer], stops_layer

@app.callback(
    Output('fuel-efficiency-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_fuel_chart(selected_route):
    route_fuel = fuel_data[fuel_data['route_id'] == selected_route]
    route_fuel['date'] = pd.to_datetime(route_fuel['date'])
    route_fuel['fuel_efficiency'] = route_fuel['distance_km'] / route_fuel['fuel_liters']
    
    fig = px.line(
        route_fuel, 
        x='date', 
        y='fuel_efficiency',
        title=f"Fuel Efficiency (km/l) for Route {selected_route}",
        labels={'fuel_efficiency': 'Fuel Efficiency (km/l)', 'date': 'Date'}
    )
    
    # Add average line
    avg_efficiency = route_fuel['fuel_efficiency'].mean()
    fig.add_hline(
        y=avg_efficiency, 
        line_dash="dot",
        annotation_text=f"Average: {avg_efficiency:.2f} km/l",
        annotation_position="bottom right"
    )
    
    return fig

@app.callback(
    Output('passenger-load-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_passenger_chart(selected_route):
    route_data = fuel_data[fuel_data['route_id'] == selected_route]
    
    fig = px.histogram(
        route_data,
        x='passenger_count',
        title=f"Passenger Load Distribution for Route {selected_route}",
        labels={'passenger_count': 'Passenger Count'},
        nbins=20
    )
    
    return fig

@app.callback(
    Output('aco-results', 'children'),
    [Input('route-selector', 'value')]
)
def update_aco_results(selected_route):
    if aco_results['best_solution']:
        return (
            f"Best Solution Score: {aco_results['best_score']:.2f}\n"
            f"Path: {' -> '.join(aco_results['best_solution']['path'])}\n"
            f"Parameters:\n"
            f"  Alpha: {aco_results['parameters']['alpha']}\n"
            f"  Beta: {aco_results['parameters']['beta']}\n"
            f"  Iterations: {aco_results['parameters']['iterations']}\n"
        )
    return "No optimization results available"

@app.callback(
    Output('route-comparison-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_comparison_chart(selected_route):
    # Calculate average fuel efficiency per route
    route_stats = fuel_data.groupby('route_id').agg({
        'distance_km': 'mean',
        'fuel_liters': 'mean',
        'passenger_count': 'mean'
    }).reset_index()
    
    route_stats['fuel_efficiency'] = (
        route_stats['distance_km'] / route_stats['fuel_liters']
    )
    
    # Sort by efficiency
    route_stats = route_stats.sort_values('fuel_efficiency', ascending=False)
    
    # Create figure
    fig = px.bar(
        route_stats,
        x='route_id',
        y='fuel_efficiency',
        color='fuel_efficiency',
        title="Fuel Efficiency Comparison Across Routes",
        labels={'fuel_efficiency': 'Fuel Efficiency (km/l)', 'route_id': 'Route ID'},
        color_continuous_scale='Viridis'
    )
    
    # Highlight selected route
    if selected_route in route_stats['route_id'].values:
        selected_idx = route_stats[route_stats['route_id'] == selected_route].index[0]
        fig.update_traces(
            marker_color=['red' if i == selected_idx else 'blue' 
                        for i in range(len(route_stats))],
            selector=dict(type='bar')
        )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)