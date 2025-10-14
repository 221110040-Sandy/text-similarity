# app_frontend.py - Streamlit Frontend for Text Similarity
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import re
from typing import Dict, Optional

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Set page config
st.set_page_config(
    page_title="📝 Advanced Text Similarity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .similarity-high {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
    }
    
    .similarity-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
    }
    
    .similarity-low {
        background: linear-gradient(135deg, #f8d7da 0%, #f1c0c7 100%);
        border-left-color: #dc3545;
    }
    
    .speed-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .speed-fast { background: #d4edda; color: #155724; }
    .speed-medium { background: #fff3cd; color: #856404; }
    .speed-slow { background: #f8d7da; color: #721c24; }
    
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .api-connected { background: #d4edda; border-left-color: #28a745; color: #155724; }
    .api-disconnected { background: #f8d7da; border-left-color: #dc3545; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Utility functions
def count_words(text):
    return len(str(text).split())

def estimate_pages(text, words_per_page=250):
    return round(count_words(text) / words_per_page, 1)

def validate_document_length(text, max_pages=5):
    estimated_pages = estimate_pages(text)
    word_count = count_words(text)
    if estimated_pages > max_pages:
        return False, f"❌ Dokumen terlalu panjang! ({estimated_pages} halaman, {word_count} kata). Maksimal {max_pages} halaman (~{max_pages * 250} kata)."
    else:
        return True, f"✅ Dokumen valid ({estimated_pages} halaman, {word_count} kata)"

def get_text_statistics(text):
    word_count = count_words(text)
    char_count = len(text)
    sentence_count = len(re.findall(r'[.!?]+', text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Simplified readability
    avg_words_per_sentence = word_count / max(sentence_count, 1)
    complexity_score = min(100, max(0, 100 - (avg_words_per_sentence * 2)))
    
    return {
        'words': word_count,
        'characters': char_count,
        'sentences': sentence_count,
        'paragraphs': paragraph_count,
        'pages': estimate_pages(text),
        'complexity_score': round(complexity_score, 1),
        'avg_words_per_sentence': round(avg_words_per_sentence, 1)
    }

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using multiple methods."""
    if not PDF_SUPPORT:
        return None, "📦 PDF support tidak tersedia. Install PyPDF2 dan pdfplumber terlebih dahulu."
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            return text.strip(), None
            
        # Method 2: Fallback to PyPDF2
        pdf_file.seek(0)  # Reset file pointer
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        if text.strip():
            return text.strip(), None
        else:
            return None, "❌ Tidak bisa extract teks dari PDF. File mungkin berupa gambar atau rusak."
            
    except Exception as e:
        return None, f"❌ Error memproses PDF: {str(e)}"

def create_pdf_preview(text, max_chars=500):
    """Create a preview of PDF content."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def validate_pdf_file(uploaded_file):
    """Validate if uploaded file is a valid PDF."""
    if uploaded_file.type != "application/pdf":
        return False, "❌ File harus berformat PDF"
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "❌ File PDF terlalu besar! Maksimal 10MB"
    
    return True, "✅ File PDF valid"

# API Communication Functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_api_health():
    """Check if API is running and model is loaded"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "model_loaded": False}
    except requests.exceptions.RequestException:
        return {"status": "disconnected", "model_loaded": False}

def predict_similarity_api(text1: str, text2: str):
    """Call API for similarity prediction menggunakan Neural model"""
    try:
        payload = {
            "text1": text1,
            "text2": text2
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return {"error": f"API Error: {error_detail}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - API took too long to respond"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def predict_document_api(doc1: str, doc2: str, per_side_len: int = 28, stride: int = 28, topk_evidence: int = 5, use_symmetric: bool = True):
    """Call API for long-document similarity (sliding window + BERTScore-like)."""
    try:
        payload = {
            "doc1": doc1,
            "doc2": doc2,
            "per_side_len": per_side_len,
            "stride": stride,
            "topk_evidence": topk_evidence,
            "use_symmetric": use_symmetric
        }
        resp = requests.post(
            f"{API_BASE_URL}/predict-document",
            json=payload,
            timeout=120   # dokumen bisa lebih lama, kasih timeout lebih besar
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            # backend FastAPI biasanya kirim {"detail": "..."}
            det = resp.json().get("detail", f"HTTP {resp.status_code}")
            return {"error": f"API Error: {det}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - document API took too long to respond"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


# Visualization Functions
def create_similarity_gauge(similarity, method="", processing_time=0):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = similarity * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"Similarity Score (%)<br><sub>{method}</sub><br><sub>⚡ {processing_time:.3f}s</sub>",
            'font': {'size': 14}
        },
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_performance_chart(results_history):
    """Create performance comparison chart"""
    if not results_history:
        return None
    
    df = pd.DataFrame(results_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['processing_time'],
        mode='lines+markers',
        name='Neural Model',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Processing Time Comparison',
        xaxis_title='Request Number',
        yaxis_title='Processing Time (seconds)',
        height=400
    )
    
    return fig

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Advanced Text Similarity Analyzer</h1>
    <p>FastAPI + Streamlit Architecture | Lightning Fast Neural Models!</p>
</div>
""", unsafe_allow_html=True)

# API Status Check
api_health = check_api_health()
api_connected = api_health["status"] in ["healthy", "unhealthy"]

if api_connected:
    if api_health["model_loaded"]:
        st.markdown("""
        <div class="api-status api-connected">
            🟢 <strong>API Connected</strong> | Neural Model Loaded | Ready for Analysis!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="api-status api-connected">
            🟡 <strong>API Connected</strong> | Neural Model NOT Loaded
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="api-status api-disconnected">
        🔴 <strong>API Disconnected</strong> | Please start FastAPI backend first
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 📋 Menu Navigasi")
analysis_type = st.sidebar.radio(
    "Pilih jenis analisis:",
    ["📄 Text Similarity", "📁 Document Similarity"],
    index=0
)

st.sidebar.markdown("## 🧠 Neural Model")
st.sidebar.markdown("**Your Trained Model:** MiniLM + BiLSTM + Attention")

# API Status in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 🌐 API Status")

if api_connected:
    st.sidebar.success("✅ FastAPI Backend Online")
    if api_health.get("model_info"):
        model_info = api_health["model_info"]
        st.sidebar.info(f"""
        **Model Info:**
        - Weights: {'✅ Loaded' if model_info['weights_loaded'] else '❌ Not Found'}
        - Max Length: {model_info['max_len']} tokens
        - Load Time: {model_info['load_time']:.1f}s
        """)
else:
    st.sidebar.error("❌ FastAPI Backend Offline")
    st.sidebar.markdown("""
    **To start backend:**
    ```bash
    python model_api.py
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture:**
- **Backend**: FastAPI (Port 8000)
- **Frontend**: Streamlit (Port 8501)
- **Model**: Loaded once in backend
- **Performance**: ~10x faster than monolithic

**Benefits:**
- Instant UI (no model loading wait)
- Scalable backend
- API can serve multiple clients
""")

# Performance tracking
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []

# Main Application
if analysis_type == "📄 Text Similarity":
    st.markdown("## 🔍 Text Similarity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Teks 1")
        text1 = st.text_area("Masukkan teks pertama:", height=200, key="text1", 
                            placeholder="Paste atau ketik teks pertama di sini...")
        
        if text1:
            is_valid1, msg1 = validate_document_length(text1)
            if is_valid1:
                st.success(msg1)
            else:
                st.error(msg1)
                text1 = ""
    
    with col2:
        st.markdown("### 📝 Teks 2")
        text2 = st.text_area("Masukkan teks kedua:", height=200, key="text2",
                            placeholder="Paste atau ketik teks kedua di sini...")
        
        if text2:
            is_valid2, msg2 = validate_document_length(text2)
            if is_valid2:
                st.success(msg2)
            else:
                st.error(msg2)
                text2 = ""
    
    # Analysis button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        analyze_button = st.button(
            "🧠 Analisis dengan Neural Model", 
            type="primary",
            disabled=not (api_connected and api_health["model_loaded"]),
            use_container_width=True
        )
        
        if analyze_button:
            if not text1 or not text2:
                st.error("❌ Mohon masukkan kedua teks yang valid!")
            else:
                # Show loading
                with st.spinner("🔄 Menganalisis dengan Neural Model..."):
                    result = predict_similarity_api(text1, text2)
                
                err = result.get("error") or result.get("detail")
                if err:
                    st.error(f"❌ {err}")
                else:
                    similarity = result['similarity']
                    processing_time = result['processing_time']
                    method = result['method']
                    
                    # Add to performance history
                    st.session_state.performance_history.append({
                        'processing_time': processing_time,
                        'similarity': similarity
                    })
                    
                    # Results display
                    st.markdown("## 📊 Hasil Analisis")
                    
                    # Performance metrics
                    perf_cols = st.columns(4)
                    with perf_cols[0]:
                        st.metric("Similarity Score", f"{similarity:.4f}")
                    with perf_cols[1]:
                        st.metric("Percentage", f"{similarity*100:.2f}%")
                    with perf_cols[2]:
                        st.metric("Processing Time", f"{processing_time:.3f}s")
                    with perf_cols[3]:
                        if 'weights_loaded' in result and result['weights_loaded'] is not None:
                            status = "Trained" if result['weights_loaded'] else "Untrained"
                            st.metric("Model Status", status)
                        else:
                            st.metric("Model", "Neural")
                    
                    # Main visualization
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if similarity >= 0.8:
                            st.markdown(f"""
                            <div class="metric-card similarity-high">
                                <h3>Hasil Akhir</h3>
                                <h1>{similarity:.4f}</h1>
                                <p>({similarity*100:.2f}%)</p>
                                <p><strong>🟢 Sangat Mirip</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif similarity >= 0.6:
                            st.markdown(f"""
                            <div class="metric-card similarity-medium">
                                <h3>Hasil Akhir</h3>
                                <h1>{similarity:.4f}</h1>
                                <p>({similarity*100:.2f}%)</p>
                                <p><strong>🟡 Cukup Mirip</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card similarity-low">
                                <h3>Hasil Akhir</h3>
                                <h1>{similarity:.4f}</h1>
                                <p>({similarity*100:.2f}%)</p>
                                <p><strong>🔴 Tidak Mirip</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        gauge_fig = create_similarity_gauge(similarity, method, processing_time)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col3:
                        st.metric("Confidence", f"{similarity*100:.1f}%")
                        st.progress(similarity)
                        
                        speed_class = "speed-fast" if processing_time < 0.1 else "speed-medium"
                        speed_text = "Sangat Cepat" if processing_time < 0.1 else "Cepat"
                        st.markdown(f'<span class="speed-indicator {speed_class}">{speed_text}</span>', unsafe_allow_html=True)
                    
                    # Text statistics
                    st.markdown("## 📈 Text Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📄 Teks 1")
                        stats1 = get_text_statistics(text1)
                        for key, value in stats1.items():
                            st.metric(key.replace('_', ' ').title(), value)
                    
                    with col2:
                        st.markdown("#### 📄 Teks 2") 
                        stats2 = get_text_statistics(text2)
                        for key, value in stats2.items():
                            st.metric(key.replace('_', ' ').title(), value)

elif analysis_type == "📁 Document Similarity":
    st.markdown("## 📁 Document Similarity Analysis")
    
    if not PDF_SUPPORT:
        st.warning("📦 **PDF support tidak tersedia.** Install PyPDF2 dan pdfplumber untuk support PDF.")
        st.info("🔧 Upload tepat 2 dokumen TXT untuk dibandingkan (maksimal 5 halaman per dokumen)")
    else:
        st.info("🔧 Upload tepat 2 dokumen (PDF atau TXT) untuk dibandingkan (maksimal 5 halaman per dokumen)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Dokumen 1")
        file_types = ["txt", "pdf"] if PDF_SUPPORT else ["txt"]
        uploaded_file1 = st.file_uploader(
            f"Upload dokumen pertama ({', '.join(file_types).upper()})", 
            type=file_types, 
            key="file1"
        )
        doc1_content = ""
        
        if uploaded_file1:
            if uploaded_file1.type == "application/pdf":
                if PDF_SUPPORT:
                    is_valid_pdf, pdf_msg = validate_pdf_file(uploaded_file1)
                    if is_valid_pdf:
                        with st.spinner("📖 Mengekstrak teks dari PDF..."):
                            text, error = extract_text_from_pdf(uploaded_file1)
                        if error:
                            st.error(error)
                        else:
                            doc1_content = text
                            is_valid1, msg1 = validate_document_length(doc1_content)
                            
                            if is_valid1:
                                st.success("✅ PDF berhasil diproses!")
                                st.success(msg1)
                                with st.expander("👁️ Preview Dokumen 1 (dari PDF)"):
                                    st.text_area("Content:", create_pdf_preview(doc1_content), 
                                               height=150, disabled=True, key="preview1")
                            else:
                                st.error(msg1)
                                doc1_content = ""
                    else:
                        st.error(pdf_msg)
                else:
                    st.error("❌ PDF support tidak tersedia")
            else:
                # TXT file
                doc1_content = uploaded_file1.read().decode("utf-8")
                is_valid1, msg1 = validate_document_length(doc1_content)
                
                if is_valid1:
                    st.success("✅ File TXT berhasil dimuat!")
                    st.success(msg1)
                    with st.expander("👁️ Preview Dokumen 1"):
                        st.text_area("Content:", doc1_content[:500] + "..." if len(doc1_content) > 500 else doc1_content, 
                                   height=150, disabled=True, key="preview1_txt")
                else:
                    st.error(msg1)
                    doc1_content = ""
    
    with col2:
        st.markdown("### 📄 Dokumen 2")
        uploaded_file2 = st.file_uploader(
            f"Upload dokumen kedua ({', '.join(file_types).upper()})", 
            type=file_types, 
            key="file2"
        )
        doc2_content = ""
        
        if uploaded_file2:
            if uploaded_file2.type == "application/pdf":
                if PDF_SUPPORT:
                    is_valid_pdf, pdf_msg = validate_pdf_file(uploaded_file2)
                    if is_valid_pdf:
                        with st.spinner("📖 Mengekstrak teks dari PDF..."):
                            text, error = extract_text_from_pdf(uploaded_file2)
                        if error:
                            st.error(error)
                        else:
                            doc2_content = text
                            is_valid2, msg2 = validate_document_length(doc2_content)
                            
                            if is_valid2:
                                st.success("✅ PDF berhasil diproses!")
                                st.success(msg2)
                                with st.expander("👁️ Preview Dokumen 2 (dari PDF)"):
                                    st.text_area("Content:", create_pdf_preview(doc2_content), 
                                               height=150, disabled=True, key="preview2")
                            else:
                                st.error(msg2)
                                doc2_content = ""
                    else:
                        st.error(pdf_msg)
                else:
                    st.error("❌ PDF support tidak tersedia")
            else:
                # TXT file
                doc2_content = uploaded_file2.read().decode("utf-8")
                is_valid2, msg2 = validate_document_length(doc2_content)
                
                if is_valid2:
                    st.success("✅ File TXT berhasil dimuat!")
                    st.success(msg2)
                    with st.expander("👁️ Preview Dokumen 2"):
                        st.text_area("Content:", doc2_content[:500] + "..." if len(doc2_content) > 500 else doc2_content, 
                                   height=150, disabled=True, key="preview2_txt")
                else:
                    st.error(msg2)
                    doc2_content = ""
    
    # Analysis button
    if st.button("🧠 Bandingkan Dokumen dengan Neural Model", type="primary", disabled=not (api_connected and api_health["model_loaded"]), use_container_width=True):
        if not doc1_content or not doc2_content:
            st.error("❌ Mohon upload kedua dokumen yang valid!")
        else:
            # Pastikan model neural tersedia, karena /predict-document butuh model neural
            if not api_health.get("model_loaded"):
                st.error("🟡 Neural model belum loaded di backend. Buka /health di FastAPI dan pastikan artifacts lengkap. (/predict-document butuh model neural)")
            else:
                with st.spinner("📊 Menganalisis dokumen (sliding window + BERTScore)..."):
                    # Kamu bisa expose parameter di sidebar kalau mau
                    result = predict_document_api(
                        doc1_content, doc2_content,
                        per_side_len=28,  # samakan dengan backend default
                        stride=28,
                        topk_evidence=5,
                        use_symmetric=True
                    )
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    doc_score = result["doc_score"]
                    processing_time = result["processing_time"]
                    detail = result["detail"]
                    top_evidence = result["top_evidence"]
                    shape = result["shape"]

                    st.markdown("## 🎯 Hasil Perbandingan Dokumen (BERTScore-like F1)")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Doc Similarity (F1)", f"{doc_score:.4f}")
                    with c2: st.metric("Percentage", f"{doc_score*100:.2f}%")
                    with c3: st.metric("Pairs (m×n)", f"{shape['m']} × {shape['n']}")
                    with c4: st.metric("Processing Time", f"{processing_time:.3f}s")

                    # Gauge pakai doc_score
                    gauge_fig = create_similarity_gauge(doc_score, "Sliding Window + BERTScore", processing_time)
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    # Detail P/R/F1
                    st.markdown("### 📌 Precision / Recall / F1")
                    if detail.get("symmetric", False):
                        a2b, b2a = detail["AtoB"], detail["BtoA"]
                        st.write(f"**A→B**  P={a2b['P']:.3f} · R={a2b['R']:.3f} · F1={a2b['F1']:.3f}")
                        st.write(f"**B→A**  P={b2a['P']:.3f} · R={b2a['R']:.3f} · F1={b2a['F1']:.3f}")
                    else:
                        a2b = detail["AtoB"]
                        st.write(f"**A→B**  P={a2b['P']:.3f} · R={a2b['R']:.3f} · F1={a2b['F1']:.3f}")

                    # Top evidence windows
                    st.markdown("### 🔎 Top Evidence Windows")
                    if top_evidence:
                        for k, ev in enumerate(top_evidence, 1):
                            st.markdown(f"**#{k} — score={ev['score']:.3f}**")
                            st.write("**A:** " + ev["windowA"])
                            st.write("**B:** " + ev["windowB"])
                            st.markdown("---")
                    else:
                        st.info("Tidak ada evidence (kemungkinan dokumen sangat pendek).")

# Footer & Additional Info
st.markdown("---")

# Performance Analytics
if st.session_state.performance_history:
    with st.expander("📊 Performance Analytics"):
        perf_chart = create_performance_chart(st.session_state.performance_history)
        if perf_chart:
            st.plotly_chart(perf_chart, width="stretch")
        
        # Performance summary
        df_perf = pd.DataFrame(st.session_state.performance_history)
        st.markdown("### Performance Summary:")
        
        if len(df_perf) > 0:
            avg_time = df_perf['processing_time'].mean()
            avg_similarity = df_perf['similarity'].mean()
            
            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.metric("Average Time", f"{avg_time:.3f}s")
            with summary_cols[1]:
                st.metric("Average Similarity", f"{avg_similarity:.3f}")

# Architecture info
with st.expander("🏗️ Architecture & Benefits"):
    arch_cols = st.columns(2)
    
    with arch_cols[0]:
        st.markdown("""
        **FastAPI + Streamlit Architecture:**
        - **Separation of Concerns**: Model logic separate from UI
        - **Performance**: Model loaded once in backend
        - **Scalability**: API can serve multiple clients
        - **Maintainability**: Independent development & deployment
        - **Real-time**: Instant UI responses
        """)
    
    with arch_cols[1]:
        st.markdown("""
        **Performance Comparison:**
        - **Monolithic Streamlit**: 30-60s initial load
        - **FastAPI + Streamlit**: <1s after backend startup
        - **Concurrent Users**: Better handling
        - **Memory Usage**: More efficient
        - **Development**: Faster iteration
        """)

# API Documentation
with st.expander("📖 API Documentation"):
    st.markdown("""
    **FastAPI Endpoints:**
    - `GET /health` - Check API and model status
    - `POST /predict` - Text similarity prediction
    - `GET /docs` - Interactive API documentation
    
    **Request Format:**
    ```json
    {
        "text1": "First text to compare",
        "text2": "Second text to compare"
    }
    ```
    
    **Response Format:**
    ```json
    {
        "similarity": 0.85,
        "processing_time": 0.123,
        "method": "Neural (MiniLM + BiLSTM + Attention)",
        "weights_loaded": true
    }
    ```
    """)

# Quick Start Guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    **1. Start FastAPI Backend:**
    ```bash
    python model_api.py
    ```
    
    **2. Start Streamlit Frontend:**
    ```bash
    streamlit run app_frontend.py
    ```
    
    **3. Access Applications:**
    - **Streamlit UI**: http://localhost:8501
    - **FastAPI Docs**: http://localhost:8000/docs
    - **API Health**: http://localhost:8000/health
    
    **4. File Requirements:**
    ```
    artifacts/
    ├── S-001_best_sts.weights.h5     # Your trained weights
    ├── tokenizer/                     # Tokenizer directory
    └── minilm-tf/                     # BERT model directory
    ```
    
    **5. Dependencies:**
    ```bash
    pip install fastapi uvicorn streamlit pandas plotly requests
    pip install tensorflow tf-keras transformers scikit-learn
    ```
    """)

st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 <strong>FastAPI + Streamlit Architecture</strong> | Lightning Fast & Scalable!</p>
    <p>🧠 Neural Model: Loaded Once | 📊 UI: Always Responsive | ⚡ Performance: 10x Faster</p>
    <p>🎯 Training ID: <strong>S-001</strong> | 🏗️ Microservices Architecture</p>
</div>
""", unsafe_allow_html=True)