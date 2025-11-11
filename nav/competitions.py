import pandas as pd
import streamlit as st
from datetime import timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.helpers import get_standard_ranking_table, load_data, get_bottom_string

#### GET DATA
df = load_data()
bottom_string = get_bottom_string()


# ---------- Leaderboard Display Function ----------
def display_leaderboard(filtered_df: pd.DataFrame, df: pd.DataFrame, selected_event: str):
    
    st.title(f"Competition: {selected_event}")
    filtered_df = filtered_df.copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date']).dt.date

    total_entrants = len(filtered_df['Name'].unique())
    fastest_time = filtered_df['time_in_seconds'].min()
    avg_time = filtered_df['time_in_seconds'].mean()

    # Calculate percentiles relative to full dataset df grouped by event:
    entrants_per_event = df.groupby('Full_Event')['Name'].nunique()
    total_entrants_percentile = entrants_per_event.rank(pct=True).loc[selected_event] * 100

    fastest_time_percentile = (df['time_in_seconds'] < fastest_time).mean() * 100

    avg_time_per_event = df.groupby('Full_Event')['time_in_seconds'].mean()
    avg_time_percentile = avg_time_per_event.rank(pct=True).loc[selected_event] * 100

    col1, col2, col3 = st.columns(3)

    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def get_delta_color(pct):
        if pct >= 75:
            return "off"
        elif pct >= 50:
            return "off"
        else:
            return "off"

    with col1:
        st.metric(
            "Total Entrants",
            total_entrants,
            delta=f"{total_entrants_percentile:.1f}%ile",
            delta_color=get_delta_color(total_entrants_percentile),
            border=True,
        )
    with col2:
        st.metric(
            "Fastest Time",
            str(timedelta(seconds=int(fastest_time))),
            delta=f"{fastest_time_percentile:.1f}%ile",
            delta_color=get_delta_color(fastest_time_percentile),
            border=True,
        )
    with col3:
        st.metric(
            "Average Time",
            str(timedelta(seconds=int(avg_time))),
            delta=f"{avg_time_percentile:.1f}%ile",
            delta_color=get_delta_color(avg_time_percentile),
            border=True,
        )

    def update_axis_layout(fig):
        fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        return fig

    def create_completion_time_plot(filtered_df: pd.DataFrame):
        RED_COLOR = '#b11f00'
        BASE_HEIGHT = 500
        BASE_LAYOUT = dict(template="plotly_white", font=dict(color="black", size=12), height=BASE_HEIGHT)

        all_df = filtered_df.copy()
        all_df['time_in_hours'] = all_df['time_in_seconds'] / 3600
        all_df.sort_values('time_in_seconds', inplace=True)
        n_all = len(all_df)
        all_df['performance'] = np.linspace(1, 0, n_all) if n_all > 1 else 1.0

        hover_template = (
            '<b>Solver:</b> %{customdata[0]}<br>'
            '<b>Rank:</b> %{customdata[4]:.0f}<br>'
            '<b>Time:</b> %{customdata[1]}<br>'
            '<b>Date:</b> %{customdata[2]|%Y-%m-%d}<br>'
            '<b>PPM:</b> %{customdata[3]:.1f}<br>'
            '<b>Performance:</b> %{x:.0%}'
        )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=all_df['performance'],
            y=all_df['time_in_hours'],
            marker=dict(color='tomato'),
            name='',
            customdata=all_df[['Name', 'Time', 'Date', 'PPM','Rank']].values,
            hovertemplate=hover_template
        ))

        tick_vals = [i/10 for i in range(0, 11)]
        tick_text = [f"{int(val*100)}%" for val in tick_vals]

        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, autorange='reversed'),
            yaxis_title='Completion Time (hours)',
            title='Completion Times (All Attempts)',
            **BASE_LAYOUT
        )

        update_axis_layout(fig)
        return fig

    def create_normalized_kde_plot(filtered_df: pd.DataFrame, df: pd.DataFrame):
        all_times = df['time_in_seconds'] / 3600
        filtered_times = filtered_df['time_in_seconds'] / 3600
    
        all_times = all_times.replace([np.inf, -np.inf], np.nan).dropna()
        filtered_times = filtered_times.replace([np.inf, -np.inf], np.nan).dropna()
    
        all_times = all_times[all_times <= 5]
        filtered_times = filtered_times[filtered_times <= 5]
    
        if len(all_times) < 2 or len(filtered_times) < 2:
            fig = go.Figure()
            fig.update_layout(
                title='KDE of Completion Times (Not enough data)',
                template='plotly_white',
                height=500
            )
            return fig
    
        kde_all = gaussian_kde(all_times)
        kde_filtered = gaussian_kde(filtered_times)
    
        x_vals = np.linspace(0, 5, 500)
    
        y_all = kde_all(x_vals)
        y_filtered = kde_filtered(x_vals)
    
        max_all = np.max(y_all)
        max_filtered = np.max(y_filtered)
        if max_filtered > 0:
            y_filtered = y_filtered * (max_all / max_filtered)
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_all,
            mode='lines',
            name='All Events',
            fill='tozeroy',
            line=dict(color='black', width=3)
        ))
    
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_filtered,
            mode='lines',
            name=filtered_df['Full_Event'].iloc[0],
            fill='tozeroy',
            line=dict(color='#1f77b4', width=3)
        ))
    
        fig.update_layout(
            xaxis_title='Completion Time (hours)',
            yaxis_title='Relative Number of Times',
            xaxis=dict(range=[0, 5]),
            yaxis=dict(range=[0, max_all * 1.05]),
            template='plotly_white',
            height=500,
            legend=dict(x=0.9, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
        )
    
        return fig

    st.plotly_chart(create_completion_time_plot(filtered_df), use_container_width=True)

    st.subheader("Leaderboard")

    view_df = filtered_df.copy()
    display_df = view_df.sort_values('Rank')[['Name', 'Time', 'Date', 'PPM','Pieces']].reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Additional Statistics")
    st.plotly_chart(create_normalized_kde_plot(filtered_df, df), use_container_width=True)
    

st.title("🏆 Competitions ")

available_events = ["Select an event"] + sorted(df['Full_Event'].unique())
selected_event = st.selectbox("Select a competition event:", available_events)

def seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=int(seconds)))

# Show leaderboard if an event is selected
if selected_event != "Select an event":
    st.session_state['selected_event'] = selected_event
    event_df = df[df['Full_Event'] == selected_event]
    display_leaderboard(event_df, df, selected_event)
else:
    # Default view: 10 most recent events average time
    st.info("Please select a competition event from the dropdown above.")
    recent_events = (
        df.dropna(subset=['Date'])
          .groupby('Full_Event')
          .agg({'time_in_seconds': 'mean', 'Date': 'max'})
          .sort_values('Date', ascending=False)
          .head(10)
          .sort_values('Date')  # chronological
          .reset_index()
    )
    recent_events['Time_HHMMSS'] = recent_events['time_in_seconds'].apply(seconds_to_hhmmss)
    
    # Use integer positions for equal spacing
    recent_events['x_pos'] = range(len(recent_events))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recent_events['x_pos'],
        y=recent_events['time_in_seconds'],
        text=recent_events['Time_HHMMSS'],
        hovertext=recent_events['Full_Event'],
        hovertemplate='<b>%{hovertext}</b><br>Avg Time: %{text}<br>Date: %{customdata}',
        customdata=recent_events['Date'].astype(str),
        marker_color='tomato'
    ))
    
    # Define y-axis ticks every 15 minutes
    y_tick_interval = 15 * 60  # 15 minutes in seconds
    y_max = recent_events['time_in_seconds'].max()
    y_ticks = list(range(0, int(y_max + y_tick_interval), y_tick_interval))
    y_tick_labels = [seconds_to_hhmmss(t) for t in y_ticks]
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=recent_events['x_pos'],
            ticktext=recent_events['Date'].astype(str),
            title='Event Date'
        ),
        yaxis=dict(
            title='Average Completion Time',
            tickmode='array',
            tickvals=y_ticks,
            ticktext=y_tick_labels
        ),
        template='plotly_white',
        height=500,
        title='Average Completion Time of the 10 Most Recent Events'
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('---')
st.markdown(bottom_string)
