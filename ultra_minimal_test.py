import streamlit as st

st.set_page_config(
    page_title="Ultra Minimal Test",
    layout="wide"
)

st.write("## Ultra Minimal Test")
st.write("If you can see this, the basic Streamlit app is working.")

# Test if we can import pandas
try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except Exception as e:
    st.error(f"❌ Pandas import failed: {e}")

# Test if we can create a simple dataframe
try:
    df = pd.DataFrame({'test': [1, 2, 3]})
    st.success("✅ DataFrame created successfully")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ DataFrame creation failed: {e}")

st.write("This is the most basic test possible.")
