import pandas as pd
import streamlit as st
from datetime import timedelta
import plotly.express as px
from utils.helpers import get_standard_ranking_table, load_data, get_bottom_string, get_jpar_display_table

#### GET DATA
df = load_data()
bottom_string = get_bottom_string()

@st.cache_data
def get_cumulative_stats(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Unique Puzzlers
    first_appearance = df.drop_duplicates(subset='Name', keep='first')
    cumulative_names = first_appearance.groupby('Date').size().cumsum().reset_index()
    cumulative_names.columns = ['Date', 'Cumulative Unique Names']

    # Total Events
    events = df.drop_duplicates(subset='Full_Event', keep='first')
    cumulative_events = events.groupby('Date').size().cumsum().reset_index()
    cumulative_events.columns = ['Date', 'Cumulative Events']

    # Total Solves
    entry_counts = df.groupby('Date').size().sort_index()
    cumulative_entries = entry_counts.cumsum().reset_index()
    cumulative_entries.columns = ['Date', 'Cumulative Times Logged']

    return cumulative_names, cumulative_events, cumulative_entries

@st.cache_data
def get_most_frequent_puzzlers(df):
    
    # get the 50 most frequent entrants to events
    frequent_puzzlers = df['Name'].value_counts().head(20)
    
    return frequent_puzzlers
 
 
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
   

# ---------- Display ----------
def display_home(df: pd.DataFrame):
    st.markdown(
        "<h1 style='text-align: center;'>🧩 USA JPA Member Dashboard </h1>",
        unsafe_allow_html=True
    )
    st.markdown('''
    Dashboard containg the competition data for USAJPA members and the JPAR Rankings! See the latest Jigsaw Puzzle Association Rating (JPAR) scores below. All of the competition data for the events going into JPAR can be accessed in the tabs on the left, as well as the specific competition data for individual puzzlers.
    ''')

    # --- Metrics block ---
    cumulative_names, cumulative_events, cumulative_entries = get_cumulative_stats(df)
    frequent_puzzlers = get_most_frequent_puzzlers(df)
    
    # --- JPAR Information ---
    st.title("📊 Jigsaw Puzzle Association Rating (JPAR)")
    
    # ---- Styled DataFrame ----
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    formatted_date = latest_date.strftime("%B %-d, %Y")  # caption still full month
    st.caption(f"Latest JPAR results containing data through {formatted_date}.")
    
    jpar_display_df = get_jpar_display_table(df)
    styled_df = jpar_display_df.style
    styled_df = styled_df.background_gradient(subset=["Latest JPAR"], cmap="Reds")
    styled_df = styled_df.background_gradient(subset=["Total Events"], cmap="Blues")
    
    # Format columns
    styled_df = styled_df.format({
        "Latest JPAR": "{:.4f}",
        "Total Events": "{:.0f}",
        "Latest Event Date": lambda x: x.strftime("%b. %-d, %Y") if pd.notnull(x) else ""
    })
    
    # Table styling
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
        
    with st.expander("🛠️ Technical JPAR Calculation", expanded=False):
        st.markdown("""
    1. **Event Mean Time**  
       For each event (defined by `event_id` and `Date`), compute the mean completion time across all participants.  
       This serves as the baseline difficulty for that event.

    2. **Event JPAR**  
       Each participant’s raw performance is normalized by dividing their completion time by the event mean time:  
       $$
       \\text{Event JPAR} = \\frac{\\text{completion time}}{\\text{event mean time}}
       $$  
       A value below 1.0 means the participant was faster than average, while above 1.0 means slower than average.

    3. **Expected Event Average (based on prior performance)**  
       If a participant has a prior JPAR from earlier events, use it to estimate how they would be expected to perform in the current event:  
       $$
       \\text{Expected Event Average} = \\text{completion time} \\times \\frac{1}{\\text{Prev JPAR}}
       $$

    4. **Mean Expected Event Average**  
       Within each event, compute the mean of all participants’ expected values (if they have one).  
       This produces an event-level adjustment factor that reflects the collective expected performance.

    5. **Adjusted Event JPAR**  
       If a mean expected average exists for the event (i.e., this event has participants with previous JPARs), normalize each participant’s raw time against it:  
       $$
       \\text{Adjusted Event JPAR} = \\frac{\\text{completion time}}{\\text{mean expected average}}
       $$  
       Otherwise, fall back to the raw Event JPAR.

    6. **JPAR Out (Running JPAR)**  
       Each participant’s ongoing JPAR is updated incrementally:  
       - If they have no history, their Adjusted JPAR is used directly.  
       - If they do have history, their new JPAR is averaged with the prior one:  
       $$
       \\text{JPAR Out} = \\frac{\\text{Prev JPAR} + \\text{Adjusted Event JPAR}}{2}
       $$

    7. **Latest JPAR**  
       At the end of processing all events, each participant’s most recent JPAR value is stored as their Latest JPAR:  
       $$
       \\text{Latest JPAR} = \\text{final JPAR Out for each participant}
       $$
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
    
    
    
display_home(df)
st.markdown('---')
st.markdown(bottom_string)


