import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('national_pbj_metrics.csv')
    return df

df = load_data()

st.title('National PBJ Metrics Dashboard')

# Create year labels for x-axis ticks
min_year = int(df['Quarter'].str[:4].min())
max_year = int(df['Quarter'].str[:4].max())
all_years = range(min_year, max_year + 1)
tick_values = [f"{year}Q1" for year in all_years]
tick_text = [str(year) for year in all_years]

# Define hover template
hover_template = "<b>%{customdata}</b><br>Value: %{y:.2f}<extra></extra>"
hover_template_count = "<b>%{customdata}</b><br>Count: %{y:,}<extra></extra>"

# 1. Total Nurse HPRD by Quarter (line chart)
st.header('Total Nurse HPRD by Quarter (2017Q1 - 2024Q4)')
df['Total_Nurse_HPRD'] = df['Total_Nurse_HPRD'].round(2)
fig1 = px.line(df, x='Quarter', y='Total_Nurse_HPRD', markers=True,
               labels={'Total_Nurse_HPRD': 'Total Nurse HPRD', 'Quarter': 'Quarter'},
               title='Total Nurse HPRD by Quarter')
fig1.update_traces(hovertemplate=hover_template, customdata=df['Quarter'])
fig1.update_xaxes(tickvals=tick_values, ticktext=tick_text, tickangle=45)
st.plotly_chart(fig1, use_container_width=True)

# 2. MDS Census by Quarter (line chart)
st.header('MDS Census by Quarter')
df['Total_MDScensus'] = df['Total_MDScensus'].round(0).astype(int)
fig2 = px.line(df, x='Quarter', y='Total_MDScensus', markers=True,
               labels={'Total_MDScensus': 'MDS Census', 'Quarter': 'Quarter'},
               title='MDS Census by Quarter')
fig2.update_traces(hovertemplate=hover_template_count, customdata=df['Quarter'])
fig2.update_xaxes(tickvals=tick_values, ticktext=tick_text, tickangle=45)
st.plotly_chart(fig2, use_container_width=True)

# 3. Contract % by Quarter (line chart)
st.header('Contract Percentage by Quarter')
df['Contract_Percentage'] = df['Contract_Percentage'].round(1)
fig3 = px.line(df, x='Quarter', y='Contract_Percentage', markers=True,
               labels={'Contract_Percentage': 'Contract %', 'Quarter': 'Quarter'},
               title='Contract Percentage by Quarter')
fig3.update_traces(hovertemplate=hover_template, customdata=df['Quarter'])
fig3.update_xaxes(tickvals=tick_values, ticktext=tick_text, tickangle=45)
st.plotly_chart(fig3, use_container_width=True)

# 4. RN HPRD by Quarter (line chart)
st.header('RN HPRD by Quarter')
df['RN_HPRD'] = df['RN_HPRD'].round(2)
fig4 = px.line(df, x='Quarter', y='RN_HPRD', markers=True,
               labels={'RN_HPRD': 'RN HPRD', 'Quarter': 'Quarter'},
               title='RN HPRD by Quarter')
fig4.update_traces(hovertemplate=hover_template, customdata=df['Quarter'])
fig4.update_xaxes(tickvals=tick_values, ticktext=tick_text, tickangle=45)
st.plotly_chart(fig4, use_container_width=True) 