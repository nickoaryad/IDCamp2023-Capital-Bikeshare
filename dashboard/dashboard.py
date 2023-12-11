'''
Author: Nicko Arya Dharma
Date: 09/12/2023
Dataset License: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
'''

#-----Import Libraries-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
import datetime
import calendar

#-----Loading Data-----
df = pd.read_csv("https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/dashboard/hour_clean.csv")
df['dteday'] = pd.to_datetime(df['dteday'])

#-----Dataframe Function-----
# Hourly riders
def create_hourly_riders_df(df):
    hourly_riders_df = df.groupby("hr").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    hourly_riders_df = hourly_riders_df.reset_index()
    hourly_riders_df.rename(columns={
        "cnt": "total_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)    
    return hourly_riders_df
# Weekday riders
def create_weekday_riders_df(df):
    weekday_riders_df = df.groupby("weekday").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    weekday_riders_df = weekday_riders_df.reset_index()
    weekday_riders_df.rename(columns={
        "cnt": "total_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)    
    return weekday_riders_df
# Monthly riders
def create_monthly_riders_df(df):
    monthly_riders_df = df.resample(rule='M', on='dteday').agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    monthly_riders_df.index = monthly_riders_df.index.strftime('%b-%y')
    monthly_riders_df = monthly_riders_df.reset_index()
    monthly_riders_df.rename(columns={
        "dteday": "month_year",
        "cnt": "total_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)    
    return monthly_riders_df
# By season riders
def create_seasonly_riders_df(df):
    seasonly_riders_df = df.groupby("season").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    seasonly_riders_df = seasonly_riders_df.reset_index()
    seasonly_riders_df.rename(columns={
        "cnt": "total_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)    
    return seasonly_riders_df
# By weather riders
def create_weatherly_riders_df(df):
    weatherly_riders_df = df.groupby("weathersit").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    weatherly_riders_df = weatherly_riders_df.reset_index()
    weatherly_riders_df.rename(columns={
        "cnt": "total_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)    
    return weatherly_riders_df

# By type of rides
def create_casual_rides_df(df):
    casual_rides_df = df.groupby("workingday").agg({
        "casual": "sum",
        "cnt": "sum"
    })
    casual_rides_df = casual_rides_df.reset_index()
    casual_rides_df.rename(columns={
        "casual": "casual_count",
        "cnt": "total_count"
    }, inplace=True)
    return casual_rides_df

def create_registered_rides_df(df):
    registered_rides_df = df.groupby("workingday").agg({
        "registered": "sum",
        "cnt": "sum"
    })
    registered_rides_df = registered_rides_df.reset_index()
    registered_rides_df.rename(columns={
        "registered": "registered_count",
        "cnt": "total_count"
    }, inplace=True)
    return registered_rides_df

#-----Filter-----
min_date = df["dteday"].min()
max_date = df["dteday"].max()

#-----Sidebar-----
with st.sidebar:
    # Add logo   
    st.markdown("[![CapitalBikeshare](https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/CapitalBikeshare_logo.jpg)](https://capitalbikeshare.com)")
    st.sidebar.header("")
    # Assign date selected
    def on_change():
        st.session_state.date = date
    # Date selector
    date = st.date_input(
        label="Select range of date", 
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        on_change=on_change
    )
    filtered_df = df[(df['dteday'] >= min_date) & (df['dteday'] <= max_date)]
    # Dataset License
    with st.expander("Dataset License"):        
        st.caption("""Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.""")
    st.sidebar.header("Thank you to:")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/IDCamp2023_logo.png")
    with col2:
        st.image("https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/Dicoding_logo.jpg") 
    st.sidebar.header("Connect with me:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/Linkedin_logo.jpg)](https://www.linkedin.com/in/nickoaryadharma)")
    with col2:
        st.markdown("[![Github](https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/GitHub_logo.jpg)](https://github.com/nickoaryad)") 

if len(date) == 2:
    main_df = df[
      (df["dteday"] >= str(date[0])) & (df["dteday"] <= str(date[1]))
    ]
else:
    main_df = df[
      (df["dteday"] >= str(st.session_state.date[0])) & (
        df["dteday"] <= str(st.session_state.date[1])
      )
    ]

# Assign dataframe function to main dataframe
hourly_riders_df = create_hourly_riders_df(main_df)
weekday_riders_df = create_weekday_riders_df(main_df)
monthly_riders_df = create_monthly_riders_df(main_df)
seasonly_riders_df = create_seasonly_riders_df(main_df)
weatherly_riders_df = create_weatherly_riders_df(main_df)
casual_rides_df = create_casual_rides_df(main_df)
registered_rides_df = create_registered_rides_df(main_df)

#-----Main Window-----
# Divide title into 2 columns contain text and logo
col1, col2 = st.columns([2,1])
with col1:
    st.title(":blue[CAPITAL] :orange[BIKESHARE]")
with col2:
    st.image("https://raw.githubusercontent.com/nickoaryad/IDCamp2023-Capital-Bikeshare/main/assets/CapitalBikeshare.jpg")

st.markdown("---")

#-----Row 1-----
_, col2, _ = st.columns([1, 2, 1])
with col2:
    st.subheader("Bike Rides Count")
# Divide subheader into 3 columns contain bike count
col1, col2, col3 = st.columns(3)
with col1:
    total_all_rides = main_df['cnt'].sum()
    st.metric(":grey[Total]", value=f'{total_all_rides:,}')
with col2:
    total_registered_rides = main_df['registered'].sum()
    st.metric(":blue[Registered]", value=f'{total_registered_rides:,}')
with col3:    
    total_casual_rides = main_df['casual'].sum()
    st.metric(":orange[Casual]", value=f'{total_casual_rides:,}')

# Create Doughnut Pie Chart
st.success("Type of Rides Distribution on Workday and Weekend/Holiday")
df['casual_percentage'] = (casual_rides_df['casual_count'] / casual_rides_df['total_count']) * 100
df['registered_percentage'] = (registered_rides_df['registered_count'] / registered_rides_df['total_count']) * 100
fig = go.Figure()
fig.add_trace(go.Pie(
    labels=registered_rides_df["workingday"],
    values=df["registered_percentage"],
    name="Year",    
    hole=0.73,
    domain=dict(x=[0, 0.45]),
))
fig.add_trace(go.Pie(
    labels=casual_rides_df['workingday'],
    values=df["casual_percentage"], 
    name='Year',    
    hole=0.73,
    domain=dict(x=[0.55, 1]),
))
fig.update_layout(
    title_text='',
    margin=dict(l=20, r=20, t=10, b=30),
    width=700, height=250,
    annotations=[
        dict(text='<b>Registered</b>', x=0.17, y=0.5, font_size=14, font_color="skyblue", showarrow=False),
        dict(text='<b>Casual</b>', x=0.81, y=0.5, font_size=14, font_color="orange", showarrow=False),
    ],
)
st.plotly_chart(fig)

#-----Row 2-----
st.success("Bike Rides Count by Timestamp")
# Draw linechart using plotly express library
fig = px.line(hourly_riders_df,
              x='hr',
              y=['casual_count', 'registered_count', 'total_count'],
              color_discrete_sequence=["orange", "skyblue", "darkgrey"],
              markers=True,
              width=700, height=350,
              title="").update_layout(xaxis_title="Hour", yaxis_title='Bike Count', margin=dict(l=20, r=20, t=10, b=30))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
st.plotly_chart(fig)

#-----Row 3-----
st.success("Bike Rides Count by Day")
# Draw linechart using plotly express library
fig = px.line(weekday_riders_df,
              x='weekday',
              y=['casual_count', 'registered_count', 'total_count'],
              color_discrete_sequence=["orange", "skyblue", "darkgrey"],              
              markers=True,
              width=700, height=350,
              title="").update_layout(xaxis_title='', yaxis_title='Bike Count', margin=dict(l=20, r=20, t=10, b=5))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
st.plotly_chart(fig)
# Draw barchart using plotly express library
fig = px.bar(weekday_riders_df,
              x='weekday',
              y=['total_count'],
              color='weekday',
              color_discrete_sequence=["orange", "powderblue", "darkgrey", "skyblue", "orangered", "slategrey", "deepskyblue"], 
              title="").update_layout(xaxis_title='', yaxis_title='Bike Count', width=700, height=350, margin=dict(l=20, r=20, t=5, b=20))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)              
st.plotly_chart(fig)

#-----Row 4-----
st.success("Bike Rides Count by Month")
# Draw linechart using plotly express library
fig = px.line(monthly_riders_df,
              x='month_year',
              y=['casual_count', 'registered_count', 'total_count'],
              color_discrete_sequence=["orange", "skyblue", "darkgrey"],
              markers=True,
              width=700, height=350,
              title="").update_layout(xaxis_title='', yaxis_title='Bike Count', margin=dict(l=20, r=20, t=10, b=5))
fig.update_xaxes(showline=True,
        linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
st.plotly_chart(fig)
# Draw barchart using plotly express library
traces = []
colors = {'2011': 'orange', '2012': 'skyblue'}
for year, color in colors.items():
    filtered_year_df = filtered_df[filtered_df['yr'] == int(year)]
    trace = go.Bar(x=filtered_year_df['dteday'].dt.month_name(),
                   y=filtered_year_df['cnt'],
                   name=f'Year {year}',
                   marker=dict(color=color)
                   )
    traces.append(trace)
layout = go.Layout(title="",
                   xaxis=dict(title='', tickangle=90),
                   yaxis=dict(title='Bike Count'),
                   barmode='stack',
                   width=700, height=350)
fig = go.Figure(data=traces, layout=layout)
fig.update_layout(showlegend=True, legend=dict(traceorder='normal'), margin=dict(l=20, r=20, t=5, b=20))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)              
st.plotly_chart(fig)

#-----Row 5-----
st.success("Bike Rides Count by Season")
# Draw linechart using plotly express library
fig = px.line(seasonly_riders_df,
              x='season',
              y=['casual_count', 'registered_count', 'total_count'],
              color_discrete_sequence=["orange", "skyblue", "darkgrey"],
              markers=True,
              width=700, height=350,
              title="").update_layout(xaxis_title='', yaxis_title='Total Count', margin=dict(l=20, r=20, t=10, b=5))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
st.plotly_chart(fig)
# Draw barchart using plotly express library
fig = px.bar(seasonly_riders_df,
              x='season',
              y=['total_count'],
              width=700, height=350,
              color='season',
              color_discrete_sequence=["orange", "skyblue", "darkgrey", "powderblue"], 
              title="").update_layout(xaxis_title='', yaxis_title='Total Count', margin=dict(l=20, r=20, t=5, b=20))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)              
st.plotly_chart(fig)

#-----Row 6-----
st.success("Bike Rides Count by Weather")
# Draw linechart using plotly express library
fig = px.line(weatherly_riders_df,
              x='weathersit',
              y=['casual_count', 'registered_count', 'total_count'],
              color_discrete_sequence=["orange", "skyblue", "darkgrey"],
              markers=True,
              width=700, height=350,
              title="").update_layout(xaxis_title='', yaxis_title='Total Count', margin=dict(l=20, r=20, t=10, b=5))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
st.plotly_chart(fig)
# Draw barchart using plotly express library
fig = px.bar(weatherly_riders_df,
              x='weathersit',
              y=['total_count'],
              color='weathersit',
              width=700, height=350,
              color_discrete_sequence=["orange", "skyblue", "darkgrey", "powderblue"], 
              title="").update_layout(xaxis_title='', yaxis_title='Total Count', margin=dict(l=20, r=20, t=5, b=20))
fig.update_xaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)
fig.update_yaxes(showline=True,
         linewidth=0.5,
         linecolor='grey',
         mirror=False)              
st.plotly_chart(fig)

# Add copyright caption
st.caption("Copyright © Nicko Arya Dharma 2023 All Rights Reserved")

# Hide Streamlit style
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)