import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="320 Admin - Subscriptions",
    page_icon="🔒",
    layout="wide"
)

# Admin password - change this to your desired password
ADMIN_PASSWORD = "320admin2024"

def get_subscriptions():
    """Get all subscriptions from the database."""
    try:
        conn = sqlite3.connect('subscriptions.db')
        query = "SELECT * FROM subscriptions ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("320 Admin - Subscriptions")
    
    # Simple password protection
    password = st.text_input("Enter admin password", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Access granted")
        
        # Get subscriptions
        df = get_subscriptions()
        
        if not df.empty:
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                entity_type = st.selectbox(
                    "Filter by Entity Type",
                    ["All"] + list(df['entity_type'].unique())
                )
            with col2:
                search_email = st.text_input("Search by email")
            
            # Apply filters
            if entity_type != "All":
                df = df[df['entity_type'] == entity_type]
            if search_email:
                df = df[df['email'].str.contains(search_email, case=False)]
            
            # Display data
            st.dataframe(df)
            
            # Export option
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "subscriptions.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.info("No subscriptions found in the database.")
    elif password:  # Only show error if password was entered
        st.error("Incorrect password")

if __name__ == "__main__":
    main() 