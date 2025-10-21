import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from utils.auth import initialize_auth_state, require_auth, is_logged_in, get_current_user, logout

st.set_page_config(
    page_title="Admin Panel",
    page_icon="",
    layout="wide"
)

initialize_auth_state()

st.markdown("""
<style>
    .admin-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

require_auth()

st.markdown("""
<div class='admin-header'>
    <h1>Admin Dashboard</h1>
    <p style='margin: 0; opacity: 0.9;'>Panel kontrol untuk mengelola sistem</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Kembali ke Home", use_container_width=True):
        st.switch_page("app_frontend.py")

with col2:
    if st.button("Logout", type="secondary", use_container_width=True):
        logout()
        st.success("Berhasil logout!")
        st.rerun()

st.markdown("---")

st.markdown("### Upload CSV File")

uploaded_file = st.file_uploader(
    "Pilih file CSV untuk diupload", 
    type=['csv'],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"File berhasil diupload: {uploaded_file.name}")
        
        st.markdown("#### Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        st.markdown("---")
        
        if st.button("Submit Data", type="primary", use_container_width=False):
            with st.spinner("Processing data..."):
                import time
                time.sleep(2)
                
                st.success("Data berhasil disubmit ke API!")
                st.info("API Response: Data processed successfully (dummy response)")
                
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")
else:
    st.info("Silakan upload file CSV untuk memulai")


