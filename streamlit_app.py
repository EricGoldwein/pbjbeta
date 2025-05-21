import streamlit as st

# Set page config BEFORE anything else runs
st.set_page_config(
    page_title="Nursing Home Staffing Dashboard (Beta)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pbjbeta import main

if __name__ == "__main__":
    main()
