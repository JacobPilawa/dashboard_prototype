import streamlit as st
from utils.helpers import get_bottom_string, get_historical_table
import pandas as pd

#### GET DATA
bottom_string = get_bottom_string()

def display_jpar_revisions():
    st.subheader("📝 Revision History")
    
    historical_table = get_historical_table()
    
    # ---- Revision History Expanders ----
    with st.expander("January 10, 2025", expanded=False):
        st.markdown("""
        **Description:** This is the description
        
        **Changes:**
        - Change A
        - Change B
        - Change C
        """)
        
        if historical_table is not None:
            st.download_button(
                label="Download JPAR from January 10, 2025",
                data=historical_table.to_csv(index=True),
                file_name="january_10_2025.csv",
                mime="text/csv"
            )
    
    with st.expander("December 15, 2024", expanded=False):
        st.markdown("""
        **Description:** This is the description
        
        **Changes:**
        - Change A
        - Change B
        - Change C
        """)
        
        if historical_table is not None:
            st.download_button(
                label="Download JPAR from December 15, 2024",
                data=historical_table.to_csv(index=True),
                file_name="december_15_2024.csv",
                mime="text/csv"
            )
    
    with st.expander("November 5, 2024", expanded=False):
        st.markdown("""
        **Description:** This is the description
        
        **Changes:**
        - Change A
        - Change B
        - Change C
        """)
        
        if historical_table is not None:
            st.download_button(
                label="Download JPAR from November 5, 2024",
                data=historical_table.to_csv(index=True),
                file_name="november_5_2024.csv",
                mime="text/csv"
            )

display_jpar_revisions()
st.markdown('---')
st.markdown(bottom_string)
