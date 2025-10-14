import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from datetime import datetime
import calendar

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variable to store the loaded data
df = None
current_provnum = None
facility_info = None

# Holiday dates for Q4 2024
holidays_2024 = {
    '2024-10-14': 'Columbus Day',
    '2024-11-11': 'Veterans Day',
    '2024-11-28': 'Thanksgiving',
    '2024-12-25': 'Christmas Day'
}

def load_q4_2024_data(provnum):
    """Load Q4 2024 data for a specific PROVNUM"""
    global df, current_provnum, facility_info
    
    # Load Q4 2024 file
    file_path = 'standardized_NonNurse/PBJ_dailynonnursestaffing_CY2024Q4.csv'
    
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    try:
        # Load the data
        df = pd.read_csv(file_path, low_memory=False)
        
        # Filter for the specific PROVNUM
        df = df[df['PROVNUM'] == provnum].copy()
        
        if df.empty:
            return f"No data found for PROVNUM {provnum} in Q4 2024"
        
        # Convert WorkDate to datetime
        df['WorkDate'] = pd.to_datetime(df['WorkDate'], format='%Y%m%d')
        
        # Add day of week
        df['DayOfWeek'] = df['WorkDate'].dt.day_name()
        df['DayOfWeekNum'] = df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Add holiday information
        df['Holiday'] = df['WorkDate'].dt.strftime('%Y-%m-%d').map(holidays_2024)
        df['IsHoliday'] = df['Holiday'].notna()
        
        # Get facility information
        facility_info = {
            'PROVNAME': df['PROVNAME'].iloc[0] if 'PROVNAME' in df.columns else 'Unknown',
            'CITY': df['CITY'].iloc[0] if 'CITY' in df.columns else 'Unknown',
            'STATE': df['STATE'].iloc[0] if 'STATE' in df.columns else 'Unknown',
            'COUNTY_NAME': df['COUNTY_NAME'].iloc[0] if 'COUNTY_NAME' in df.columns else 'Unknown'
        }
        
        current_provnum = provnum
        return f"Successfully loaded {len(df)} records for PROVNUM {provnum}"
        
    except Exception as e:
        return f"Error loading data: {str(e)}"

# Define the layout with enhanced styling
app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.H1("🏥 Nursing Home Staffing Portal", 
                    style={'color': '#2c3e50', 'margin': '0', 'fontSize': '2.5rem', 'fontWeight': 'bold'}),
            html.P("Non-Nurse Staffing Analysis Dashboard - Q4 2024", 
                   style={'color': '#7f8c8d', 'margin': '5px 0 0 0', 'fontSize': '1.1rem'})
        ], style={'textAlign': 'center', 'padding': '20px 0'})
    ], style={'backgroundColor': '#ecf0f1', 'borderBottom': '3px solid #3498db', 'marginBottom': '30px'}),
    
    # Main Container
    html.Div([
        # Facility Selection Section
        html.Div([
            html.Div([
                html.H2("📍 Facility Selection", 
                        style={'color': '#2c3e50', 'marginBottom': '20px', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                html.P("Enter the Provider Number (PROVNUM) for the nursing home you want to analyze:", 
                       style={'color': '#34495e', 'fontSize': '16px', 'marginBottom': '15px'}),
                html.Div([
                    dcc.Input(
                        id='provnum-input',
                        type='text',
                        placeholder='Enter PROVNUM (e.g., 015009)',
                        value='',
                        style={'width': '250px', 'marginRight': '15px', 'padding': '12px', 
                               'border': '2px solid #bdc3c7', 'borderRadius': '8px', 'fontSize': '16px'}
                    ),
                    html.Button('🔍 Load Facility Data', id='load-button', n_clicks=0,
                               style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 
                                      'padding': '12px 24px', 'cursor': 'pointer', 'borderRadius': '8px',
                                      'fontSize': '16px', 'fontWeight': 'bold', 'transition': 'all 0.3s ease'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                html.Div(id='load-status', style={'fontWeight': 'bold', 'fontSize': '16px'})
            ], style={'padding': '25px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'})
        ]),
        
        # Facility Information Section
        html.Div(id='facility-info', style={'display': 'none'}),
        
        # Main Analysis Section
        html.Div(id='analysis-section', style={'display': 'none'}, children=[
            # Staffing Overview Section
            html.Div([
                html.H2("📊 Staffing Overview", 
                        style={'color': '#2c3e50', 'marginBottom': '20px', 'borderBottom': '2px solid #27ae60', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Label("Select Staff Category:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px'}),
                        dcc.Dropdown(
                            id='staff-category-dropdown',
                            options=[
                                {'label': '👨‍⚕️ Medical Director', 'value': 'Hrs_MedDir'},
                                {'label': '👔 Administrator', 'value': 'Hrs_Admin'},
                                {'label': '👨‍⚕️ Other MD', 'value': 'Hrs_OthMD'},
                                {'label': '👨‍⚕️ Physician Assistant', 'value': 'Hrs_PA'},
                                {'label': '👩‍⚕️ Nurse Practitioner', 'value': 'Hrs_NP'},
                                {'label': '👩‍⚕️ Clinical Nurse Specialist', 'value': 'Hrs_ClinNrsSpec'},
                                {'label': '💊 Pharmacist', 'value': 'Hrs_Pharmacist'},
                                {'label': '🥗 Dietician', 'value': 'Hrs_Dietician'},
                                {'label': '🍽️ Feeding Assistant', 'value': 'Hrs_FeedAsst'},
                                {'label': '🏥 Occupational Therapist', 'value': 'Hrs_OT'},
                                {'label': '🏥 Physical Therapist', 'value': 'Hrs_PT'},
                                {'label': '🫁 Respiratory Therapist', 'value': 'Hrs_RespTher'},
                                {'label': '🗣️ Speech Language Pathologist', 'value': 'Hrs_SpcLangPath'},
                                {'label': '🤝 Social Worker', 'value': 'Hrs_QualSocWrk'},
                                {'label': '🧠 Mental Health Services', 'value': 'Hrs_MHSvc'}
                            ],
                            value='Hrs_MedDir',
                            style={'width': '300px'}
                        )
                    ], style={'margin': '15px', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginBottom': '30px'}),
                
                # Analysis Results
                html.Div(id='analysis-results'),
                
                # Charts Grid
                html.Div([
                    html.Div([
                        dcc.Graph(id='daily-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    html.Div([
                        dcc.Graph(id='day-of-week-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='holiday-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    html.Div([
                        dcc.Graph(id='summary-stats-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
                ])
            ], style={'padding': '25px', 'backgroundColor': 'white', 'borderRadius': '10px', 
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'})
        ]),
        
        # Threshold Analysis Section (Separate and Distinct)
        html.Div(id='threshold-section', style={'display': 'none'}, children=[
            html.Div([
                html.H2("⚠️ Compliance Threshold Analysis", 
                        style={'color': '#e74c3c', 'marginBottom': '20px', 'borderBottom': '3px solid #e74c3c', 'paddingBottom': '10px'}),
                html.P("This section analyzes staffing levels against compliance thresholds to identify potential regulatory issues:", 
                       style={'color': '#34495e', 'fontSize': '16px', 'marginBottom': '20px', 'fontStyle': 'italic'}),
                
                html.Div([
                    html.Div([
                        html.Label("Staff Category:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px'}),
                        dcc.Dropdown(
                            id='threshold-staff-dropdown',
                            options=[
                                {'label': '👨‍⚕️ Medical Director', 'value': 'Hrs_MedDir'},
                                {'label': '👔 Administrator', 'value': 'Hrs_Admin'},
                                {'label': '👨‍⚕️ Other MD', 'value': 'Hrs_OthMD'},
                                {'label': '👨‍⚕️ Physician Assistant', 'value': 'Hrs_PA'},
                                {'label': '👩‍⚕️ Nurse Practitioner', 'value': 'Hrs_NP'},
                                {'label': '👩‍⚕️ Clinical Nurse Specialist', 'value': 'Hrs_ClinNrsSpec'},
                                {'label': '💊 Pharmacist', 'value': 'Hrs_Pharmacist'},
                                {'label': '🥗 Dietician', 'value': 'Hrs_Dietician'},
                                {'label': '🍽️ Feeding Assistant', 'value': 'Hrs_FeedAsst'},
                                {'label': '🏥 Occupational Therapist', 'value': 'Hrs_OT'},
                                {'label': '🏥 Physical Therapist', 'value': 'Hrs_PT'},
                                {'label': '🫁 Respiratory Therapist', 'value': 'Hrs_RespTher'},
                                {'label': '🗣️ Speech Language Pathologist', 'value': 'Hrs_SpcLangPath'},
                                {'label': '🤝 Social Worker', 'value': 'Hrs_QualSocWrk'},
                                {'label': '🧠 Mental Health Services', 'value': 'Hrs_MHSvc'}
                            ],
                            value='Hrs_MedDir',
                            style={'width': '300px'}
                        )
                    ], style={'margin': '15px', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.Label("Compliance Threshold (hours):", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '8px'}),
                        dcc.Input(
                            id='threshold-input',
                            type='number',
                            placeholder='Enter threshold (e.g., 8)',
                            value=8,
                            style={'width': '150px', 'padding': '8px', 'border': '2px solid #e74c3c', 
                                   'borderRadius': '5px', 'fontSize': '16px'}
                        )
                    ], style={'margin': '15px', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginBottom': '25px'}),
                
                # Threshold Analysis Results
                html.Div(id='threshold-results'),
                
                # Threshold Charts
                html.Div([
                    html.Div([
                        dcc.Graph(id='threshold-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    html.Div([
                        dcc.Graph(id='threshold-timeline', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
                ])
            ], style={'padding': '25px', 'backgroundColor': '#fff5f5', 'borderRadius': '10px', 
                      'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'border': '2px solid #e74c3c'})
        ])
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 20px'})
])

# Callback to load data
@app.callback(
    Output('load-status', 'children'),
    Output('analysis-section', 'style'),
    Output('threshold-section', 'style'),
    Output('facility-info', 'children'),
    Output('facility-info', 'style'),
    Input('load-button', 'n_clicks'),
    State('provnum-input', 'value'),
    prevent_initial_call=True
)
def load_data_callback(n_clicks, provnum):
    if not provnum:
        return "Please enter a PROVNUM", {'display': 'none'}, {'display': 'none'}, "", {'display': 'none'}
    
    status = load_q4_2024_data(provnum)
    
    if "Successfully" in status:
        # Create facility info display
        facility_display = html.Div([
            html.H2("🏢 Facility Information", 
                    style={'color': '#2c3e50', 'marginBottom': '20px', 'borderBottom': '2px solid #27ae60', 'paddingBottom': '10px'}),
            html.Div([
                html.Div([
                    html.H4("🏥 Provider Details", style={'color': '#27ae60', 'marginBottom': '15px'}),
                    html.P(f"Provider Name: {facility_info['PROVNAME']}", 
                           style={'margin': '8px 0', 'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"Provider Number: {current_provnum}", 
                           style={'margin': '8px 0', 'fontSize': '16px', 'color': '#7f8c8d'})
                ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '50%', 'padding': '15px'}),
                
                html.Div([
                    html.H4("📍 Location", style={'color': '#27ae60', 'marginBottom': '15px'}),
                    html.P(f"City: {facility_info['CITY']}", 
                           style={'margin': '8px 0', 'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"State: {facility_info['STATE']}", 
                           style={'margin': '8px 0', 'fontSize': '16px'}),
                    html.P(f"County: {facility_info['COUNTY_NAME']}", 
                           style={'margin': '8px 0', 'fontSize': '16px'})
                ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '50%', 'padding': '15px'})
            ])
        ], style={'margin': '20px 0', 'padding': '25px', 'border': '2px solid #27ae60', 
                  'borderRadius': '10px', 'backgroundColor': '#d5f4e6', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        
        return status, {'display': 'block'}, {'display': 'block'}, facility_display, {'display': 'block'}
    else:
        return status, {'display': 'none'}, {'display': 'none'}, "", {'display': 'none'}

# Callback to update main analysis results
@app.callback(
    Output('analysis-results', 'children'),
    Output('daily-chart', 'figure'),
    Output('day-of-week-chart', 'figure'),
    Output('holiday-chart', 'figure'),
    Output('summary-stats-chart', 'figure'),
    Input('staff-category-dropdown', 'value'),
    prevent_initial_call=True
)
def update_analysis(staff_category):
    global df
    
    if df is None or df.empty:
        return "No data loaded", {}, {}, {}, {}
    
    if staff_category not in df.columns:
        return f"Column {staff_category} not found in data", {}, {}, {}, {}
    
    # Calculate statistics
    total_days = len(df)
    avg_hours = df[staff_category].mean()
    min_hours = df[staff_category].min()
    max_hours = df[staff_category].max()
    std_hours = df[staff_category].std()
    
    # Day of week analysis
    dow_stats = df.groupby('DayOfWeek')[staff_category].agg(['mean', 'count']).reset_index()
    dow_stats = dow_stats.sort_values('mean', ascending=False)
    
    # Holiday analysis
    holiday_stats = df.groupby('IsHoliday')[staff_category].agg(['mean', 'count']).reset_index()
    holiday_stats['Category'] = holiday_stats['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})
    
    # Create results text
    results_text = html.Div([
        html.H3(f"📈 {staff_category} Analysis Summary", 
                style={'color': '#2c3e50', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("📊 Key Metrics", style={'color': '#3498db', 'marginBottom': '15px'}),
                html.P(f"Total days analyzed: {total_days}", 
                       style={'fontSize': '16px', 'margin': '8px 0'}),
                html.P(f"Average hours: {avg_hours:.2f}", 
                       style={'fontSize': '16px', 'margin': '8px 0', 'fontWeight': 'bold'}),
                html.P(f"Range: {min_hours:.2f} - {max_hours:.2f} hours", 
                       style={'fontSize': '16px', 'margin': '8px 0'}),
                html.P(f"Standard deviation: {std_hours:.2f}", 
                       style={'fontSize': '16px', 'margin': '8px 0'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '50%', 'padding': '15px'})
        ], style={'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '20px'})
    ], style={'margin': '20px 0', 'padding': '20px', 'border': '1px solid #bdc3c7', 
              'borderRadius': '8px', 'backgroundColor': 'white'})
    
    # Create daily chart
    daily_fig = px.line(df, x='WorkDate', y=staff_category, 
                       title=f'📅 Daily {staff_category} Hours - Q4 2024')
    daily_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Hours",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create day of week chart
    dow_fig = px.bar(dow_stats, x='DayOfWeek', y='mean', 
                    title=f'📊 Average {staff_category} Hours by Day of Week',
                    color='mean', color_continuous_scale='viridis')
    dow_fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Hours",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create holiday comparison chart
    holiday_fig = px.bar(holiday_stats, x='Category', y='mean', 
                        title=f'🎉 {staff_category} Hours: Holiday vs Non-Holiday',
                        color='Category', color_discrete_map={'Holiday': '#e74c3c', 'Non-Holiday': '#3498db'})
    holiday_fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Average Hours",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create summary stats chart
    summary_fig = go.Figure()
    summary_fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=avg_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Average {staff_category} Hours"},
        delta={'reference': avg_hours},
        gauge={
            'axis': {'range': [None, max_hours * 1.1]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, min_hours], 'color': "lightgray"},
                {'range': [min_hours, avg_hours], 'color': "#d5f4e6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_hours
            }
        }
    ))
    summary_fig.update_layout(
        title=f"📊 {staff_category} Performance Gauge",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return results_text, daily_fig, dow_fig, holiday_fig, summary_fig

# Callback to update threshold analysis
@app.callback(
    Output('threshold-results', 'children'),
    Output('threshold-chart', 'figure'),
    Output('threshold-timeline', 'figure'),
    Input('threshold-staff-dropdown', 'value'),
    Input('threshold-input', 'value'),
    prevent_initial_call=True
)
def update_threshold_analysis(staff_category, threshold):
    global df
    
    if df is None or df.empty:
        return "No data loaded", {}, {}
    
    if staff_category not in df.columns:
        return f"Column {staff_category} not found in data", {}, {}
    
    # Calculate threshold statistics
    total_days = len(df)
    days_below_threshold = len(df[df[staff_category] < threshold])
    days_above_threshold = total_days - days_below_threshold
    percentage_below = (days_below_threshold/total_days*100)
    
    # Get specific days below threshold
    below_threshold_days = df[df[staff_category] < threshold][['WorkDate', staff_category, 'DayOfWeek', 'Holiday']].copy()
    below_threshold_days['WorkDate'] = below_threshold_days['WorkDate'].dt.strftime('%m-%d-%Y')
    
    # Create threshold results text
    threshold_text = html.Div([
        html.H3(f"⚠️ Compliance Analysis for {staff_category}", 
                style={'color': '#e74c3c', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("🚨 Compliance Status", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                html.P(f"Days below {threshold} hours: {days_below_threshold}", 
                       style={'fontSize': '18px', 'margin': '8px 0', 'fontWeight': 'bold', 
                              'color': '#e74c3c' if days_below_threshold > 0 else '#27ae60'}),
                html.P(f"Days above {threshold} hours: {days_above_threshold}", 
                       style={'fontSize': '16px', 'margin': '8px 0'}),
                html.P(f"Compliance rate: {(100-percentage_below):.1f}%", 
                       style={'fontSize': '16px', 'margin': '8px 0', 'fontWeight': 'bold'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '50%', 'padding': '15px'}),
            
            html.Div([
                html.H4("📋 Non-Compliant Days", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                html.Div([
                    html.Ul([html.Li(f"{row['WorkDate']} ({row['DayOfWeek']})" + 
                                   (f" - {row['Holiday']}" if pd.notna(row['Holiday']) else "") + 
                                   f": {row[staff_category]:.2f} hours") 
                            for _, row in below_threshold_days.iterrows()])
                ]) if not below_threshold_days.empty else html.P("✅ No non-compliant days found!", 
                                                                 style={'color': '#27ae60', 'fontWeight': 'bold'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '50%', 'padding': '15px'})
        ])
    ], style={'margin': '20px 0', 'padding': '20px', 'border': '2px solid #e74c3c', 
              'borderRadius': '8px', 'backgroundColor': '#fff5f5'})
    
    # Create threshold analysis chart
    threshold_fig = go.Figure()
    threshold_fig.add_trace(go.Bar(
        x=['Below Threshold', 'Above Threshold'],
        y=[days_below_threshold, days_above_threshold],
        text=[days_below_threshold, days_above_threshold],
        textposition='auto',
        marker_color=['#e74c3c', '#27ae60']
    ))
    threshold_fig.update_layout(
        title=f'⚠️ Days Below vs Above {threshold} Hours Threshold',
        xaxis_title='Category',
        yaxis_title='Number of Days',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create threshold timeline chart
    timeline_fig = px.scatter(df, x='WorkDate', y=staff_category, 
                             title=f'📅 {staff_category} Hours vs Threshold Timeline',
                             color=df[staff_category] < threshold,
                             color_discrete_map={True: '#e74c3c', False: '#27ae60'},
                             labels={'color': 'Below Threshold'})
    timeline_fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                          annotation_text=f"Threshold: {threshold} hours")
    timeline_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Hours",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return threshold_text, threshold_fig, timeline_fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8050) 