import base64
import streamlit as st

def pbj_icon(size: int = 20, margin_right: int = 8):
    """
    Display the PBJ favicon as an icon in your Streamlit app.
    
    Parameters:
    -----------
    size : int
        Size of the icon in pixels (default: 20)
    margin_right : int
        Right margin in pixels (default: 8)
    """
    try:
        # Read and encode the PBJ favicon
        with open('pbj_favicon.png', 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode()
        
        # Create the HTML for the icon
        html = f"""
        <img src="data:image/png;base64,{encoded_image}" 
             style="width: {size}px; height: {size}px; margin-right: {margin_right}px;">
        """
        
        st.markdown(html, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error("PBJ favicon not found. Please ensure 'pbj_favicon.png' is in the current directory.")
    except Exception as e:
        st.error(f"Error loading PBJ icon: {str(e)}")

def pbj_icon_with_text(text: str, size: int = 20, margin_right: int = 8):
    """
    Display the PBJ favicon with text in a flex container.
    
    Parameters:
    -----------
    text : str
        Text to display next to the icon
    size : int
        Size of the icon in pixels (default: 20)
    margin_right : int
        Right margin in pixels (default: 8)
    """
    try:
        # Read and encode the PBJ favicon
        with open('pbj_favicon.png', 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode()
        
        # Create the HTML for the icon with text
        html = f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{encoded_image}" 
                 style="width: {size}px; height: {size}px; margin-right: {margin_right}px;">
            <span>{text}</span>
        </div>
        """
        
        st.markdown(html, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error("PBJ favicon not found. Please ensure 'pbj_favicon.png' is in the current directory.")
    except Exception as e:
        st.error(f"Error loading PBJ icon: {str(e)}")
