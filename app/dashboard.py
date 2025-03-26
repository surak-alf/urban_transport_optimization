import dash
from dash import dcc, html, Input, Output
import dash_leaflet as dl
import plotly.express as px
import pandas as pd
import networkx as nx
import pickle
from datetime import datetime

# Load and validate data
def load_data():
    """Load all required data with validation"""
    data = {}
    
    try:
        with open('data/processed/transport_graph.pkl', 'rb') as f:
            data['graph'] = pickle.load(f)
        
        with open('data/processed/aco_results.pkl', 'rb') as f:
            data['aco_results'] = pickle.load(f)
        
        data['bus_stops'] = pd.read_csv('data/raw/bus_stops.csv')
        data['bus_routes'] = pd.read_csv('data/raw/bus_routes.csv')
        data['fuel_data'] = pd.read_csv('data/raw/fuel_consumption.csv')
        
        # Convert route stops from string to list
        data['bus_routes']['stops'] = data['bus_routes']['stops'].apply(
            lambda x: x.strip("[]").replace("'", "").split(", "))
        
        return data
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

def verify_data_consistency(data):
    """Check that all route stops exist in bus_stops"""
    missing_stops = set()
    for _, route in data['bus_routes'].iterrows():
        for stop in route['stops']:
            if stop not in data['bus_stops']['stop_id'].values:
                missing_stops.add(stop)
    
    if missing_stops:
        print(f"Warning: {len(missing_stops)} stops in routes are missing from bus_stops:")
        print(missing_stops)
    else:
        print("Data consistency verified - all route stops exist in bus_stops data")

# Load data
try:
    data = load_data()
    verify_data_consistency(data)
    
    graph = data['graph']
    aco_results = data['aco_results']
    bus_stops = data['bus_stops']
    bus_routes = data['bus_routes']
    fuel_data = data['fuel_data']
    
except Exception as e:
    print(f"Fatal error during data loading: {str(e)}")
    raise

# Create Dash app
app = dash.Dash(__name__, title="Transport Optimization Dashboard",
               suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Urban Public Transport Optimization Dashboard", 
           style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='route-selector',
                options=[{'label': f"Route {i+1} ({route_id})", 'value': route_id} 
                        for i, route_id in enumerate(bus_routes['route_id'])],
                value=bus_routes['route_id'].iloc[0],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dl.Map(
                id='map',
                center=[bus_stops['latitude'].mean(), bus_stops['longitude'].mean()],
                zoom=12,
                style={'height': '500px', 'width': '100%', 'border': '1px solid #ddd'},
                children=[
                    dl.TileLayer(),
                    dl.LayerGroup(id='route-layer'),
                    dl.LayerGroup(id='stops-layer')
                ]
            )
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='fuel-efficiency-chart',
                     style={'height': '250px', 'marginBottom': '20px'}),
            dcc.Graph(id='passenger-load-chart',
                     style={'height': '250px', 'marginBottom': '20px'}),
            html.Div([
                html.H3("ACO Optimization Results", style={'marginTop': '20px'}),
                html.Div(id='aco-results',
                        style={'backgroundColor': '#f8f9fa', 
                              'padding': '10px',
                              'borderRadius': '5px',
                              'fontFamily': 'monospace'})
            ])
        ], style={'width': '38%', 'display': 'inline-block', 
                 'verticalAlign': 'top', 'padding': '10px'})
    ], style={'marginBottom': '20px'}),
    
    html.Div([
        dcc.Graph(id='route-comparison-chart',
                 style={'height': '400px'})
    ], style={'padding': '10px', 'borderTop': '1px solid #eee'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})

@app.callback(
    [Output('route-layer', 'children'),
     Output('stops-layer', 'children')],
    [Input('route-selector', 'value')]
)
def update_map(selected_route):
    try:
        # Get selected route data
        route = bus_routes[bus_routes['route_id'] == selected_route].iloc[0]
        stops = route['stops']
        
        # Create route line
        route_line = []
        valid_stops = []
        
        for i in range(len(stops)-1):
            try:
                start = bus_stops[bus_stops['stop_id'] == stops[i]].iloc[0]
                end = bus_stops[bus_stops['stop_id'] == stops[i+1]].iloc[0]
                
                route_line.append([
                    [start['longitude'], start['latitude']],
                    [end['longitude'], end['latitude']]
                ])
                valid_stops.extend([stops[i], stops[i+1]])
            except IndexError:
                continue
        
        # Create route layer
        route_layer = dl.Polyline(
            positions=route_line,
            color='#1f77b4',
            weight=4,
            opacity=0.8
        ) if route_line else None
        
        # Create stops layer with only valid stops
        stops_layer = []
        unique_stops = list(set(valid_stops))  # Remove duplicates
        
        for stop_id in unique_stops:
            try:
                stop = bus_stops[bus_stops['stop_id'] == stop_id].iloc[0]
                is_terminal = graph.nodes[stop_id].get('is_terminal', False)
                
                stops_layer.append(
                    dl.CircleMarker(
                        center=[stop['latitude'], stop['longitude']],
                        radius=10 if is_terminal else 8,
                        color='#2ca02c' if is_terminal else '#d62728',
                        fill=True,
                        fillOpacity=0.8,
                        children=dl.Tooltip(
                            f"{stop['stop_name']} {'(Terminal)' if is_terminal else ''}"
                        )
                    )
                )
            except IndexError:
                continue
        
        return [route_layer] if route_layer else [], stops_layer
    
    except Exception as e:
        print(f"Error updating map: {str(e)}")
        return [], []

@app.callback(
    Output('fuel-efficiency-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_fuel_chart(selected_route):
    try:
        route_fuel = fuel_data[fuel_data['route_id'] == selected_route].copy()
        route_fuel['date'] = pd.to_datetime(route_fuel['date'])
        route_fuel['fuel_efficiency'] = route_fuel['distance_km'] / route_fuel['fuel_liters']
        
        fig = px.line(
            route_fuel, 
            x='date', 
            y='fuel_efficiency',
            title=f"Fuel Efficiency (km/l) - Route {selected_route}",
            labels={'fuel_efficiency': 'Fuel Efficiency (km/l)', 'date': 'Date'},
            template='plotly_white'
        )
        
        # Add average line
        avg_efficiency = route_fuel['fuel_efficiency'].mean()
        fig.add_hline(
            y=avg_efficiency, 
            line_dash="dot",
            line_color="red",
            annotation_text=f"Average: {avg_efficiency:.2f} km/l",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            hovermode='closest'
        )
        
        return fig
    except Exception as e:
        print(f"Error updating fuel chart: {str(e)}")
        return px.line(title="Data not available")

@app.callback(
    Output('passenger-load-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_passenger_chart(selected_route):
    try:
        route_data = fuel_data[fuel_data['route_id'] == selected_route]
        
        fig = px.histogram(
            route_data,
            x='passenger_count',
            title=f"Passenger Load Distribution - Route {selected_route}",
            labels={'passenger_count': 'Passenger Count'},
            nbins=15,
            color_discrete_sequence=['#9467bd'],
            template='plotly_white'
        )
        
        fig.update_layout(
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            showlegend=False
        )
        
        return fig
    except Exception as e:
        print(f"Error updating passenger chart: {str(e)}")
        return px.histogram(title="Data not available")

@app.callback(
    Output('aco-results', 'children'),
    [Input('route-selector', 'value')]
)
def update_aco_results(selected_route):
    try:
        if aco_results and 'best_solution' in aco_results:
            return [
                html.P(f"Best Solution Score: {aco_results['best_score']:.2f}"),
                html.P(f"Optimized Path: {' → '.join(aco_results['best_solution']['path'])}"),
                html.Hr(),
                html.P("ACO Parameters:"),
                html.P(f"• Alpha (pheromone): {aco_results['parameters']['alpha']}"),
                html.P(f"• Beta (heuristic): {aco_results['parameters']['beta']}"),
                html.P(f"• Iterations: {aco_results['parameters']['iterations']}"),
                html.P(f"• Ants: {aco_results['parameters']['num_ants']}")
            ]
        return "No optimization results available"
    except Exception as e:
        print(f"Error updating ACO results: {str(e)}")
        return "Error loading optimization results"

@app.callback(
    Output('route-comparison-chart', 'figure'),
    [Input('route-selector', 'value')]
)
def update_comparison_chart(selected_route):
    try:
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
        route_stats = route_stats.sort_values('fuel_efficiency', ascending=True)
        
        # Create figure
        fig = px.bar(
            route_stats,
            y='route_id',
            x='fuel_efficiency',
            orientation='h',
            title="Fuel Efficiency Comparison Across All Routes",
            labels={'fuel_efficiency': 'Fuel Efficiency (km/l)', 'route_id': 'Route ID'},
            color='fuel_efficiency',
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        
        # Highlight selected route
        if selected_route in route_stats['route_id'].values:
            selected_idx = route_stats[route_stats['route_id'] == selected_route].index[0]
            fig.update_traces(
                marker_color=['#ef553b' if i == selected_idx else '#636efa' 
                            for i in range(len(route_stats))],
                selector=dict(type='bar')
            )
        
        fig.update_layout(
            margin={'l': 100, 'b': 40, 't': 60, 'r': 10},
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception as e:
        print(f"Error updating comparison chart: {str(e)}")
        return px.bar(title="Data not available")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)