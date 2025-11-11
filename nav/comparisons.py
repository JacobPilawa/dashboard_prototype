import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import get_standard_ranking_table, load_data, get_bottom_string
from utils.helpers import prepare_jpar_trends, get_person_jpar_trajectory

#### GET DATA
df = load_data()
bottom_string = get_bottom_string()

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"
    
    

def display_summary_stats(comparison_df, df, selected_puzzlers):
    """
    Display summary statistics for selected puzzlers, including:
      - Summary table
      - Avg time bar chart
      - JPAR over time plot with percentile bands and individual puzzler lines
    """

    st.markdown("""Here's some summary information for the selected puzzlers, sorted by their average time over the last 12 months.""")

    # --- SUMMARY TABLE ---
    summary_df = comparison_df[['Name', 'Time', 'PPM', 'time_in_seconds', 'Date', 'Latest JPAR']].copy()
    summary_df['Date'] = pd.to_datetime(summary_df['Date'], errors='coerce')

    # Filter last 12 months
    latest_date = summary_df['Date'].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)
    recent_df = summary_df[summary_df['Date'] >= one_year_ago]

    grouped_summary = (
        summary_df.groupby('Name')
        .agg(
            avg_time_in_seconds=('time_in_seconds', 'mean'),
            avg_ppm=('PPM', 'mean'),
            count=('Name', 'size')
        )
        .reset_index()
    )

    recent_summary = (
        recent_df.groupby('Name')
        .agg(
            last_year_avg_time=('time_in_seconds', 'mean'),
            last_year_avg_ppm=('PPM', 'mean'),
            last_year_count=('Name', 'size')
        )
        .reset_index()
    )

    grouped_summary = grouped_summary.merge(recent_summary, on='Name', how='left')

    latest_jpar = (
        summary_df.sort_values("Date")
        .groupby("Name")
        .tail(1)[["Name", "Latest JPAR"]]
    )

    grouped_summary = grouped_summary.merge(latest_jpar, on="Name", how="left")

    grouped_summary['Average Time'] = grouped_summary['avg_time_in_seconds'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}"
    )
    grouped_summary['Average Time (Last 12 Months)'] = grouped_summary['last_year_avg_time'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}" if pd.notnull(x) else "-"
    )
    grouped_summary['Average PPM'] = grouped_summary['avg_ppm'].round(2)
    grouped_summary['Average PPM (Last 12 Months)'] = grouped_summary['last_year_avg_ppm'].round(2).fillna("-")
    grouped_summary['Number of Events'] = grouped_summary['count']
    grouped_summary['Number of Events (Last 12 Months)'] = grouped_summary['last_year_count'].fillna(0).astype(int)

    final_summary = grouped_summary[[
        'Name',
        'Number of Events',
        'Average Time',
        'Latest JPAR'
    ]]

    st.dataframe(final_summary.reset_index(drop=True), use_container_width=True)

    # --- BAR CHART OF AVG TIME (LAST 12 MONTHS) ---
    plot_df = grouped_summary.dropna(subset=['last_year_avg_time']).copy()
    plot_df = plot_df.sort_values(by='last_year_avg_time', ascending=True)
    plot_df['time_in_hours'] = plot_df['last_year_avg_time'] / 3600
    plot_df['avg_time_hms'] = plot_df['last_year_avg_time'].apply(seconds_to_hms)

    hover_template = (
        '<b>Solver:</b> %{x}<br>'
        '<b>Avg Time (Last 12 Months):</b> %{customdata[0]}<br>'
        '<b>PPM:</b> %{customdata[1]:.2f}<br>'
        '<b>Events:</b> %{customdata[2]:.0f}<br>'
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df['Name'],
        y=plot_df['time_in_hours'],
        marker=dict(color='tomato'),
        customdata=plot_df[['avg_time_hms', 'last_year_avg_ppm', 'last_year_count']].values,
        hovertemplate=hover_template
    ))

    fig.update_layout(
        title='Average Completion Time (Last 12 Months)',
        xaxis_title='Solver',
        yaxis_title='Avg Completion Time (hours)',
        template='plotly_white',
        font=dict(color='black', size=12),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- JPAR OVER TIME WITH PERCENTILE BANDS ---
    st.subheader("📈 JPAR Over Time 📉")

    trend_df = prepare_jpar_trends(df)
    if not trend_df.empty:
        fig_jpar = go.Figure()

        # Percentile bands
        bands = [
            ("p20", "p80", "rgba(220, 50, 50, 0.25)", "20–80% Range"),
            ("p30", "p70", "rgba(220, 50, 50, 0.40)", "30–70% Range"),
            ("p40", "p60", "rgba(220, 50, 50, 0.60)", "40–60% Range"),
        ]
        for low, high, color, label in bands:
            fig_jpar.add_trace(go.Scatter(
                x=pd.concat([trend_df["event_date"], trend_df["event_date"][::-1]]),
                y=pd.concat([trend_df[low], trend_df[high][::-1]]),
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                hoverinfo="skip",
                name=label,
                showlegend=True
            ))

        # Median line
        fig_jpar.add_trace(go.Scatter(
            x=trend_df["event_date"],
            y=trend_df["median"],
            mode="lines",
            line=dict(color="darkred", width=2),
            marker=dict(size=0),
            name="Median JPAR"
        ))
        
        # Choose a continuous sequential colorscale
        colorscale = pc.sequential.Rainbow  # or any other Plotly sequential scale
        
        # Generate evenly spaced colors for the number of puzzlers
        num_puzzlers = len(selected_puzzlers)
        color_indices = np.linspace(0, 0.85, num_puzzlers)  # evenly spaced between 0 and 1
        colors = [pc.sample_colorscale(colorscale, idx)[0] for idx in color_indices]
        
        # Assign a color to each puzzler
        for i, name in enumerate(selected_puzzlers):
            person_traj = get_person_jpar_trajectory(df, name)
            if not person_traj.empty:
                fig_jpar.add_trace(go.Scatter(
                    x=person_traj["event_date"],
                    y=person_traj["JPAR Out"],
                    mode="lines+markers",
                    line=dict(width=5, color=colors[i]),
                    marker=dict(size=10),
                    name=name
                ))
                
        # Optional: Show fastest puzzler per event
        show_fastest = st.checkbox("Show fastest puzzler per event", value=False)
        if show_fastest:
            fig_jpar.add_trace(go.Scatter(
                x=trend_df["event_date"],
                y=trend_df["fastest"],
                mode="lines+markers",
                line=dict(color="black", width=2),
                name="Fastest JPAR"
            ))

        # Layout
        fig_jpar.update_layout(
            title="Evolution of JPAR Distribution Over Time",
            xaxis_title="Event Date",
            yaxis_title="JPAR",
            template="plotly_white",
            hovermode="x unified",
            height=500,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="gray",
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=40, b=20)
        )


        st.plotly_chart(fig_jpar, use_container_width=True)
    else:
        st.info("No JPAR trend data available to plot.")

def display_all_stats(comparison_df, df):
    st.markdown("""Here's all the information for all of the puzzlers selected above, sorted first by name and then by date of event.""")

    # filter only the columns we want
    display_df = comparison_df.sort_values('Date')[['Name','Date', 'Full_Event', 'Rank', 
    'Time', 'PPM', 'Remaining', 'JPAR Out']].copy()

    # rename some columns
    display_df = display_df.rename(columns={'Full_Event': 'Event Name'})

    # convert to datetime first
    display_df['Date'] = pd.to_datetime(display_df['Date'], errors='coerce')
    display_df['Date'] = display_df['Date'].dt.date

    # Calculate total entrants per event
    event_totals = df.groupby('Full_Event')['Name'].count()

    # Create Rank column as "N/T"
    display_df['Rank'] = display_df.apply(lambda row: f"{int(row['Rank'])}/{event_totals.get(row['Event Name'], 0)}", axis=1)

    # Sort by most recent
    display_df = display_df.sort_values(by=['Name', 'Date'], ascending=False)

    # --- Bar Plot of All Times ---
    plot_df = comparison_df[['Name', 'time_in_seconds', 'Full_Event']].copy()
    plot_df = plot_df.sort_values(by='time_in_seconds')  # Fastest first
    plot_df['time_in_hours'] = plot_df['time_in_seconds'] / 3600
    
    # Create unique x labels for consistent spacing
    plot_df['label'] = [f'{n} {i}' for i, n in enumerate(plot_df['Name'])]
    
    # Assign one consistent color per puzzler
    color_seq = px.colors.qualitative.Plotly
    seen = {}
    ordered_names = []
    
    # Keep track of order of appearance
    for name in plot_df['Name']:
        if name not in seen:
            seen[name] = True
            ordered_names.append(name)
    
    color_map = {name: color_seq[i % len(color_seq)] for i, name in enumerate(ordered_names)}
    plot_df['color'] = plot_df['Name'].map(color_map)
    plot_df['time_in_hms'] = plot_df['time_in_seconds'].apply(seconds_to_hms)
    
    
    # Bar plot using go.Bar
    hover_template = (
        '<b>Solver:</b> %{customdata[0]}<br>'
        '<b>Time:</b> %{customdata[2]}<br>'
        '<b>Event:</b> %{customdata[1]}<br>'
        '<extra></extra>'
    )
    
    fig = go.Figure()
    
    # Single trace for all bars sorted by time
    fig.add_trace(go.Bar(
        x=plot_df['label'],
        y=plot_df['time_in_hours'],
        marker=dict(color=plot_df['color']),
        customdata=plot_df[['Name', 'Full_Event', 'time_in_hms']],
        hovertemplate=hover_template,
        showlegend=False
    ))
    
    # Add dummy traces for legend entries per person
    for name in ordered_names:
        fig.add_trace(go.Bar(
            x=[None], y=[None],  # Dummy invisible bar for legend only
            name=name,
            marker=dict(color=color_map[name]),
            showlegend=True
        ))
    
    fig.update_layout(
        title='All Completion Times (Sorted by Fastest)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(title='Completion Time (hours)'),
        template='plotly_white',
        font=dict(size=12),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Final Table ---
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
    


def display_comparison(df):
    
    # all names
    puzzler_names = sorted(df['Name'].dropna().unique())
    
    # names currently selected
    selected_puzzlers = st.multiselect(
        "Select puzzlers to compare:",
        sorted(df['Name'].dropna().unique()),
    )

    if len(selected_puzzlers) < 2:
        st.info("Please select at least two puzzlers.")
    else:
        comparison_df = df[df['Name'].isin(selected_puzzlers)]
        st.subheader("📊 Summary Statistics")
        display_summary_stats(comparison_df, df, selected_puzzlers)

        st.subheader("📄 All Events")
        display_all_stats(comparison_df, df)
    
        

    
st.title("⚔️ Compare Puzzlers")
display_comparison(df)
st.markdown('---')
st.markdown(bottom_string)

