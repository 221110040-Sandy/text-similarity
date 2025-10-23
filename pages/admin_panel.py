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
    <p style='margin: 0; opacity: 0.9;'>Panel kontrol untuk training data</p>
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

st.markdown("### Upload CSV Files (Train / Val / Test)")

col_train, col_val, col_test = st.columns(3)

train_file = None
val_file = None
test_file = None

with col_train:
    st.markdown("**Train CSV**")
    train_file = st.file_uploader("Upload train CSV", type=['csv'], key='train_file')
    if train_file:
        try:
            df_train = pd.read_csv(train_file)
            st.success(f"Train file loaded: {getattr(train_file, 'name', 'train')}")
            st.dataframe(df_train.head(5), use_container_width=True)
            st.metric("Rows", len(df_train))
            st.metric("Columns", len(df_train.columns))
        except Exception as e:
            st.error(f"Error membaca train file: {str(e)}")

with col_val:
    st.markdown("**Validation CSV**")
    val_file = st.file_uploader("Upload val CSV", type=['csv'], key='val_file')
    if val_file:
        try:
            df_val = pd.read_csv(val_file)
            st.success(f"Val file loaded: {getattr(val_file, 'name', 'val')}")
            st.dataframe(df_val.head(5), use_container_width=True)
            st.metric("Rows", len(df_val))
            st.metric("Columns", len(df_val.columns))
        except Exception as e:
            st.error(f"Error membaca val file: {str(e)}")

with col_test:
    st.markdown("**Test CSV**")
    test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test_file')
    if test_file:
        try:
            df_test = pd.read_csv(test_file)
            st.success(f"Test file loaded: {getattr(test_file, 'name', 'test')}")
            st.dataframe(df_test.head(5), use_container_width=True)
            st.metric("Rows", len(df_test))
            st.metric("Columns", len(df_test.columns))
        except Exception as e:
            st.error(f"Error membaca test file: {str(e)}")

st.markdown("---")

start_col1 = st.columns(1)[0]
with start_col1:
    if st.button("Mulai Training", type="primary", use_container_width=True):
        # It's okay if training doesn't actually run; show dummy behavior
        if not train_file and not val_file and not test_file:
            st.warning("Upload semua file untuk mulai training.")
        else:
            with st.spinner("Memulai training (dummy)..."):
                import time
                time.sleep(2)
            st.success("Training started (dummy).")
            uploaded_names = [getattr(f, 'name', '') for f in [train_file, val_file, test_file] if f]
            st.info(f"Uploaded files: {', '.join(uploaded_names)}")


