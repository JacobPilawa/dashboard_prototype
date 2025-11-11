import streamlit as st
from utils.helpers import get_ranking_table, load_data, get_bottom_string, get_standard_ranking_table
import pandas as pd
import plotly.express as px

#### GET DATA
df = load_data()
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

    return jpar_display_df[['Name', 'Latest JPAR', 'Total Events', 'Latest Event Date', 'Latest Event']]
    
def display_jpar_ratings(df):
    st.title("📊 Jigsaw Puzzle Association Rating (JPAR)")
    
    # ---- Styled DataFrame ----
    jpar_display_df = get_jpar_display_table(df)
    styled_df = jpar_display_df.style
    styled_df = styled_df.background_gradient(subset=["Latest JPAR"], cmap="Reds")
    styled_df = styled_df.background_gradient(subset=["Total Events"], cmap="Blues")
    styled_df = styled_df.format({
        "Latest JPAR": "{:.4f}",
        "Latest Event Date": lambda x: x.strftime("%d/%m/%Y") if pd.notnull(x) else ""
    })
    styled_df = styled_df.set_table_styles([
        {"selector": "thead th", "props": [("background-color", "#f4f4f4"), ("font-weight", "bold")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fafafa")]},
    ])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # ---- Explanatory Text in Expanders ----
    with st.expander("❓ What is JPAR?", expanded=False):
        st.markdown("""
        JPAR is a calculation of your puzzling performance compared to the average USAJPA member over the last 12 months.

        - **Faster than average:** Below 1  
        - **Slower than average:** Above 1
        """)

    with st.expander("⚙️ How does it work?", expanded=False):
        st.markdown("""
        **For each event:**
        1. **We compile:**  
           - Completion Times for all USAJPA members who competed  
           - Pre-existing JPAR for all competitors
        2. **We calculate:**  
           - Expected Completion Times by multiplying each member’s Completion Time by their inverse JPAR, then averaging for an Expected Event Average  
           - Event JPAR = Completion Time / Expected Event Average
        3. **Final JPAR:** Average previous JPAR with the new Event JPAR
        """)

    with st.expander("📌 Important Notes", expanded=False):
        st.markdown("""
        - If a member does not finish within the max time, an estimated time is calculated.  
        - First-time competitors:
            - Do not contribute to Expected Event Average  
            - Event JPAR becomes their final JPAR  
        - Expected Event Average ensures fair ranking when many participants are new  
        - Puzzle difficulty is considered relative to other competitors  
        - Only USAJPA member results are used
        """)
        
    st.caption("For more information, visit: [usajigsaw.org/jpar](https://www.usajigsaw.org/jpar)")

    # --------- Medal Counts Plot ---------
    st.subheader("🏅 Medal Counts")
    top_medalists_df = get_top_medalists(df)
    long_df = top_medalists_df.melt(
        id_vars="Name",
        value_vars=["Gold", "Silver", "Bronze"],
        var_name="Medal Type",
        value_name="Number of Medals",
    )

    fig = px.bar(
        long_df,
        x="Name",
        y="Number of Medals",
        color="Medal Type",
        title="Top Medal Winners",
        labels={"Name": "Puzzler"},
        color_discrete_map={"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32"},
    )
    fig.update_layout(barmode="stack", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

display_jpar_ratings(df)
st.markdown('---')
st.markdown(bottom_string)
