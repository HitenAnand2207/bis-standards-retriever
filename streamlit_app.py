import streamlit as st
import json
from src.bis_retriever import BisRetriever
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="BIS Standards Retriever",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .result-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin-top: 6px;
    }
    .metric-card {
        background: rgba(102, 126, 234, 0.10);
        border: 1px solid rgba(102, 126, 234, 0.18);
        border-radius: 10px;
        padding: 10px 8px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-bottom: 4px;
        line-height: 1.1;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 0.68rem;
        color: #86efac;
        margin-top: 4px;
        line-height: 1.1;
    }
    .header-title {
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = "dataset.pdf"

# Load retriever
@st.cache_resource
def load_retriever(pdf_path):
    if Path(pdf_path).exists():
        return BisRetriever(pdf_path)
    return None

# Main title
st.markdown("<div class='header-title'>📚 BIS Standards Retrieval System</div>", unsafe_allow_html=True)
st.markdown("*Fast, accurate retrieval of Indian Standards with intelligent ranking*")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Top-k selection
    top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=5)
    
    st.divider()
    st.subheader("📊 Performance Metrics")
    st.markdown(
        """
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Hit@3</div>
                <div class="metric-value">100%</div>
                <div class="metric-delta">+30%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MRR@5</div>
                <div class="metric-value">0.8167</div>
                <div class="metric-delta">+0.117</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-value">0.03s</div>
                <div class="metric-delta">Fast</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load test queries
test_set_path = Path("data/public_test_set.json")
try:
    with open(test_set_path) as f:
        test_data = json.load(f)
    test_queries = {q["id"]: q["query"] for q in test_data}
except:
    test_queries = {}

# Load retriever
retriever = load_retriever(st.session_state.pdf_path)

if retriever is None:
    st.error(f"❌ Error: Could not load PDF from '{st.session_state.pdf_path}'")
    st.info("Make sure `dataset.pdf` exists in the project root directory.")
else:
    # Main interface
    tab1, tab2 = st.tabs(["🔍 Query", "📋 Test Set Results"])
    
    # Tab 1: Query Interface
    with tab1:
        st.subheader("Enter a Query")
        
        # Quick examples
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "What BIS standard are you looking for?",
                placeholder="e.g., What is ordinary portland cement?",
                label_visibility="collapsed"
            )
        with col2:
            search_clicked = st.button("🔍 Search", use_container_width=True)
        
        # Display results
        if search_clicked and query:
            with st.spinner("Searching..."):
                import time
                start = time.time()
                results = retriever.retrieve(query, top_k=top_k)
                latency = time.time() - start
            
            st.success(f"Found {len(results)} results in {latency:.3f} seconds")
            st.divider()
            
            for i, result in enumerate(results, 1):
                if isinstance(result, tuple) and len(result) >= 2:
                    standard_id, score = result[0], result[1]
                else:
                    standard_id, score = result, None
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {i}. {standard_id}")
                    with col2:
                        if score is not None:
                            st.metric("Score", f"{score:.3f}")
                        else:
                            st.metric("Rank", f"#{i}")
                    
                    st.divider()
        
        elif search_clicked:
            st.warning("Please enter a query")
    
    # Tab 2: Test Set Results
    with tab2:
        st.subheader("Public Test Set Performance")
        
        # Load submission results
        submission_path = Path("data/submission.json")
        if submission_path.exists():
            with open(submission_path) as f:
                submission = json.load(f)
            
            # Load ground truth
            ground_truth = {}
            with open(test_set_path) as f:
                for item in json.load(f):
                    ground_truth[item["id"]] = item["expected_standards"]
            
            # Display results
            st.info(f"📊 Showing results from {len(submission)} queries")
            st.divider()
            
            for query_result in submission:
                query_id = query_result["id"]
                retrieved = query_result["retrieved_standards"]
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**Query {query_id}**")
                    if query_id in test_queries:
                        st.caption(test_queries[query_id])
                
                with col2:
                    if query_id in ground_truth:
                        gt = ground_truth[query_id][0] if ground_truth[query_id] else "N/A"
                        is_hit = gt in retrieved[:3]
                        status = "✅ Hit" if is_hit else "❌ Miss"
                        st.markdown(f"**Truth:** {gt} {status}")
                
                with col3:
                    st.metric("Retrieved", len(retrieved))
                
                # Show retrieved standards
                with st.expander(f"View {len(retrieved)} results"):
                    for rank, std in enumerate(retrieved, 1):
                        st.text(f"{rank}. {std}")
                
                st.divider()
        else:
            st.warning("No submission results found. Run `python inference.py` first.")

