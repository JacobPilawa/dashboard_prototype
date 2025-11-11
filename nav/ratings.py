import streamlit as st
from utils.helpers import get_ranking_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_ranking_table, load_data, get_bottom_string, get_standard_ranking_table

#### GET DATA
df = load_data()
styled_table, results = get_standard_ranking_table()
bottom_string = get_bottom_string()

# ---------- Plotting Utilities ----------
@st.cache_data
def get_top_medalists(df, top_n=20):
    medal_counts = df[df['Rank'].isin([1, 2, 3])].copy()
    medal_counts['Gold'] = (medal_counts['Rank'] == 1).astype(int)
    medal_counts['Silver'] = (medal_counts['Rank'] == 2).astype(int)
    medal_counts['Bronze'] = (medal_counts['Rank'] == 3).astype(int)

    grouped = medal_counts.groupby('Name')[['Gold', 'Silver', 'Bronze']].sum()
    grouped['Total'] = grouped.sum(axis=1)
    top = grouped.sort_values('Total', ascending=False).head(top_n).drop(columns='Total')
    
    return top.reset_index()
    
@st.cache_data
def get_jpar_display_table(df):
    df['Latest Event Date'] = pd.to_datetime(df['Latest Event Date'])
    df_sorted = df.sort_values('Latest Event Date', ascending=False)
    jpar_display_df = df_sorted.drop_duplicates(subset='member_id', keep='first')
    jpar_display_df = jpar_display_df.sort_values(by='Latest JPAR').reset_index(drop=True)
    jpar_display_df.index += 1
    jpar_display_df['Latest Event Date'] = jpar_display_df['Latest Event Date'].dt.date
    jpar_display_df = jpar_display_df[['Name', 'Latest JPAR', 'Total Events', 'Latest Event Date', 'Latest Event']]
    return jpar_display_df
    
# ---------- JPAR Ratings Display Function ----------
def display_jpar_ratings(styled_table, results, df):
    
    st.title("📊 Jigsaw Puzzle Association Rating (JPAR)")
    st.markdown(f"More information: [usajigsaw.org/jpar](https://www.usajigsaw.org/jpar)")

    # --- Explanatory Section ---
    with st.expander("❓ What is JPAR?"):
        st.markdown("""
        **What is JPAR?**  
        A calculation of your puzzling performance compared to average USAJPA member performance across each USAJPA/sanctioned competition within the last 12 months.
        
        - Faster than average = Below 1  
        - Slower than average = Above 1
        """)
    
    with st.expander("⚙️ How does it work?"):
        st.markdown("""
        **For each event:**  
        1. **We compile:**  
           - Completion Times for all USAJPA members who competed.  
           - Pre-existing JPAR for all USAJPA members who competed.
        2. **We calculate:**  
           - Expected Completion Times for each competitor by multiplying each member’s Completion Time by their inverse JPAR and averaging them for an Expected Event Average.  
           - Event JPAR for each member by dividing their Completion Time by the Expected Event Average.  
        3. **Final JPAR:** Average previous JPAR with new Event JPAR for each member.
        """)
    
    with st.expander("📌 Important Notes"):
        st.markdown("""
        - If a member does not complete their puzzle within the maximum time limit, an estimated completion time is calculated based on their puzzling rate (pieces/time limit).  
        - If competing for the first time:
            - Their time does not contribute to the Expected Event Average.  
            - Their Event JPAR is not averaged, but serves as their final JPAR.  
        - The Expected Event Average ensures fair ranking even when many competitors are new.  
        - Puzzle difficulty is considered by comparing times relative to other participants.  
        - Only USAJPA member results are used in JPAR calculations.
        """)

    # --- Display JPAR Table ---
    jpar_display_df = get_jpar_display_table(df)
    st.dataframe(jpar_display_df, use_container_width=True)

    # --- Medal Counts Plot ---
    st.subheader("🏅 Medal Counts")
    top_medalists_df = get_top_medalists(df)
    long_df = top_medalists_df.melt(
        id_vars='Name', 
        value_vars=['Gold', 'Silver', 'Bronze'],
        var_name='Medal Type',
        value_name='Number of Medals'
    )
    
    fig = px.bar(
        long_df,
        x='Name',
        y='Number of Medals',
        color='Medal Type',
        title="Top Medal Winners",
        labels={"Name": "Puzzler"},
        color_discrete_map={"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32"},
    )
    fig.update_layout(barmode='stack', xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

display_jpar_ratings(styled_table, results, df)

st.markdown('---')
st.markdown(bottom_string)
