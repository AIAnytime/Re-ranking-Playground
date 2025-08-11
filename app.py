import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import cohere
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Re-ranking Playground",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'reranking_results' not in st.session_state:
        st.session_state.reranking_results = {}
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_documents' not in st.session_state:
        st.session_state.current_documents = []

# Load API keys and models
@st.cache_resource
def load_models():
    """Load and cache re-ranking models"""
    models = {}
    try:
        # Try different cross-encoder model names
        try:
            models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except:
            try:
                models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
            except:
                st.warning("Cross-encoder model not available. Some features may be limited.")
        
        # Load Cohere API
        cohere_key = os.getenv('COHERE_API_KEY')
        if cohere_key and cohere_key.strip():
            try:
                models['cohere'] = cohere.ClientV2(api_key=cohere_key)
                print("✅ Cohere API client loaded successfully")
            except Exception as e:
                st.warning(f"Cohere API connection failed: {str(e)}")
        else:
            st.warning("Cohere API key not found in environment")
        
        # Load OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key.strip():
            try:
                models['openai'] = OpenAI(api_key=openai_key)
                print("✅ OpenAI API client loaded successfully")
            except Exception as e:
                st.warning(f"OpenAI API setup failed: {str(e)}")
        else:
            st.warning("OpenAI API key not found in environment")
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models

# Sample data
SAMPLE_QUERIES = {
    "Technology": {
        "query": "artificial intelligence machine learning applications",
        "documents": [
            "Deep learning neural networks for computer vision applications",
            "Machine learning algorithms in healthcare diagnostics", 
            "Artificial intelligence applications in autonomous vehicles",
            "Traditional statistical methods for data analysis",
            "Blockchain technology and cryptocurrency trading",
            "Natural language processing for sentiment analysis",
            "Computer graphics and 3D modeling techniques",
            "Quantum computing and quantum algorithms"
        ]
    },
    "Science": {
        "query": "climate change environmental impact research",
        "documents": [
            "Global warming effects on polar ice caps",
            "Carbon dioxide emissions from industrial processes",
            "Renewable energy sources and sustainability",
            "Ocean acidification and marine ecosystems",
            "Deforestation impact on biodiversity",
            "Solar panel efficiency and cost analysis",
            "Weather pattern changes due to climate shift",
            "Electric vehicle adoption and environmental benefits"
        ]
    },
    "Business": {
        "query": "digital marketing strategy customer engagement",
        "documents": [
            "Social media marketing campaigns and ROI analysis",
            "Customer relationship management systems",
            "Digital advertising platforms and targeting",
            "E-commerce conversion optimization strategies",
            "Brand awareness through content marketing",
            "Email marketing automation and personalization",
            "Search engine optimization best practices",
            "Influencer marketing in the digital age"
        ]
    },
    "Healthcare": {
        "query": "medical diagnosis artificial intelligence",
        "documents": [
            "AI-powered medical imaging for cancer detection",
            "Machine learning in drug discovery and development",
            "Electronic health records and patient data analysis",
            "Telemedicine platforms and remote patient monitoring",
            "Precision medicine and personalized treatment plans",
            "Medical chatbots and virtual health assistants",
            "Predictive analytics for hospital resource management",
            "Clinical decision support systems using AI"
        ]
    }
}

# Re-ranking techniques
class ReRankingTechniques:
    def __init__(self, models):
        self.models = models
    
    def cross_encoder_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Cross-encoder re-ranking implementation"""
        try:
            if 'cross_encoder' not in self.models:
                raise Exception("Cross-encoder model not available")
            
            pairs = [(query, doc) for doc in documents]
            scores = self.models['cross_encoder'].predict(pairs)
            
            results = [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1,
                    'technique': 'Cross-Encoder'
                }
                for i, (doc, score) in enumerate(
                    sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
                )
            ]
            
            return results
            
        except Exception as e:
            st.error(f"Cross-encoder error: {str(e)}")
            return []
    
    def cohere_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Cohere API re-ranking"""
        try:
            if 'cohere' not in self.models:
                raise Exception("Cohere API not available")
            
            response = self.models['cohere'].rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            results = []
            for i, result in enumerate(response.results):
                # Get the original document text using the index
                doc_index = result.index
                doc_text = documents[doc_index] if doc_index < len(documents) else "Unknown document"
                
                results.append({
                    'document': doc_text,
                    'score': result.relevance_score,
                    'rank': i + 1,
                    'technique': 'Cohere API'
                })
            
            return results
            
        except Exception as e:
            st.error(f"Cohere re-ranking error: {str(e)}")
            return []
    
    def llm_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """LLM-based re-ranking using OpenAI GPT"""
        try:
            openai_client = self.models.get('openai')
            if not openai_client:
                raise Exception("OpenAI API not available")
            
            docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
            
            prompt = f"""
            Query: "{query}"
            
            Documents to rank:
            {docs_text}
            
            Please rank these documents by relevance to the query. Return only the document numbers in order of relevance (most relevant first), separated by commas.
            Example response: 3,1,5,2,4
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            
            ranking_str = response.choices[0].message.content.strip()
            ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
            
            results = []
            for i, doc_idx in enumerate(ranking_indices[:top_k]):
                if 0 <= doc_idx < len(documents):
                    results.append({
                        'document': documents[doc_idx],
                        'score': 1.0 - (i * 0.1),
                        'rank': i + 1,
                        'technique': 'LLM-based'
                    })
            
            return results
            
        except Exception as e:
            st.error(f"LLM re-ranking error: {str(e)}")
            return []
    
    def hybrid_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Hybrid re-ranking combining multiple techniques"""
        try:
            cross_results = self.cross_encoder_rerank(query, documents, len(documents))
            
            vectorizer = TfidfVectorizer()
            all_texts = [query] + documents
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            combined_scores = []
            for i, doc in enumerate(documents):
                cross_score = cross_results[i]['score'] if i < len(cross_results) else 0
                tfidf_score = similarities[i]
                combined_score = 0.7 * cross_score + 0.3 * tfidf_score
                combined_scores.append((doc, combined_score))
            
            sorted_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
            
            results = [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1,
                    'technique': 'Hybrid'
                }
                for i, (doc, score) in enumerate(sorted_results)
            ]
            
            return results
            
        except Exception as e:
            st.error(f"Hybrid re-ranking error: {str(e)}")
            return []
    
    def learning_to_rank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Learning to Rank simulation"""
        try:
            features = []
            query_words = set(query.lower().split())
            
            for doc in documents:
                doc_words = set(doc.lower().split())
                
                jaccard_sim = len(query_words & doc_words) / len(query_words | doc_words) if query_words | doc_words else 0
                doc_length = len(doc.split())
                query_coverage = len(query_words & doc_words) / len(query_words) if query_words else 0
                
                features.append([jaccard_sim, doc_length / 100, query_coverage])
            
            weights = [0.5, 0.2, 0.3]
            scores = [sum(w * f for w, f in zip(weights, feat)) for feat in features]
            
            sorted_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
            
            results = [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1,
                    'technique': 'Learning to Rank'
                }
                for i, (doc, score) in enumerate(sorted_results)
            ]
            
            return results
            
        except Exception as e:
            st.error(f"Learning to Rank error: {str(e)}")
            return []

def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔄 Re-ranking Playground")
        st.markdown("---")
        
        st.markdown("### Built with ❤️")
        st.markdown("[AI Anytime](https://aianytime.net)")
        st.markdown("[Sonu Kumar](https://sonukumar.site)")
        st.markdown("---")
        
        st.markdown("### Model Status")
        models = load_models()
        
        if 'cross_encoder' in models:
            st.markdown("✅ Cross-Encoder Ready")
        else:
            st.markdown("❌ Cross-Encoder Failed")
            
        if 'cohere' in models:
            st.markdown("✅ Cohere API Ready")
        else:
            st.markdown("❌ Cohere API Failed")
            
        if models.get('openai'):
            st.markdown("✅ OpenAI API Ready")
        else:
            st.markdown("❌ OpenAI API Failed")
    
    # Main header
    st.markdown('<h1 class="main-header">Re-ranking Playground</h1>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Playground", 
        "📊 Techniques", 
        "🔬 Evaluation", 
        "📚 Case Studies",
        "ℹ️ About"
    ])
    
    with tab1:
        playground_tab()
    
    with tab2:
        techniques_tab()
    
    with tab3:
        evaluation_tab()
    
    with tab4:
        case_studies_tab()
    
    with tab5:
        about_tab()

def playground_tab():
    """Main playground interface"""
    st.markdown("## 🎯 Interactive Re-ranking Playground")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Query Input")
        query = st.text_input(
            "Enter your search query:",
            value=st.session_state.current_query,
            placeholder="e.g., machine learning applications in healthcare"
        )
        
        st.markdown("### Quick Start with Presets")
        preset_category = st.selectbox("Choose a preset:", ["Custom", "Technology", "Science", "Business", "Healthcare"])
        
        if preset_category != "Custom":
            if st.button(f"Load {preset_category} Preset"):
                preset_data = SAMPLE_QUERIES[preset_category]
                st.session_state.current_query = preset_data["query"]
                st.session_state.current_documents = preset_data["documents"]
                st.rerun()
    
    with col2:
        st.markdown("### Settings")
        top_k = st.slider("Results to return:", 1, 15, 8)
        techniques = st.multiselect(
            "Select techniques:",
            ["Cross-Encoder", "LLM-based", "Cohere API", "Hybrid", "Learning to Rank"],
            default=["Cross-Encoder", "Cohere API"]
        )
    
    st.markdown("### Documents to Re-rank")
    documents_text = st.text_area(
        "Enter documents (one per line):",
        value="\n".join(st.session_state.current_documents),
        height=200
    )
    
    documents = [doc.strip() for doc in documents_text.split('\n') if doc.strip()]
    
    if st.button("🚀 Start Re-ranking", type="primary"):
        if query and documents:
            st.session_state.current_query = query
            st.session_state.current_documents = documents
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models = load_models()
            reranker = ReRankingTechniques(models)
            results = {}
            
            for i, technique in enumerate(techniques):
                status_text.markdown(f'<div class="status-processing">Processing: {technique}...</div>', unsafe_allow_html=True)
                
                try:
                    if technique == "Cross-Encoder":
                        results[technique] = reranker.cross_encoder_rerank(query, documents, top_k)
                    elif technique == "LLM-based":
                        results[technique] = reranker.llm_rerank(query, documents, top_k)
                    elif technique == "Cohere API":
                        results[technique] = reranker.cohere_rerank(query, documents, top_k)
                    elif technique == "Hybrid":
                        results[technique] = reranker.hybrid_rerank(query, documents, top_k)
                    elif technique == "Learning to Rank":
                        results[technique] = reranker.learning_to_rank(query, documents, top_k)
                    
                    progress_bar.progress((i + 1) / len(techniques))
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"Error with {technique}: {str(e)}")
                    results[technique] = []
            
            status_text.markdown('<div class="status-success">✅ Re-ranking completed!</div>', unsafe_allow_html=True)
            st.session_state.reranking_results = results
            
            display_results(results, query)
        else:
            st.warning("Please enter both query and documents.")

def display_results(results: Dict, query: str):
    """Display re-ranking results"""
    st.markdown("## 📊 Results")
    
    if results:
        technique_tabs = st.tabs(list(results.keys()))
        
        for i, (technique, technique_results) in enumerate(results.items()):
            with technique_tabs[i]:
                if technique_results:
                    for result in technique_results:
                        col1, col2, col3 = st.columns([1, 6, 1])
                        with col1:
                            st.markdown(f"**#{result['rank']}**")
                        with col2:
                            st.markdown(f"{result['document']}")
                        with col3:
                            st.markdown(f"**{result['score']:.3f}**")
                else:
                    st.warning(f"No results for {technique}")
        
        # Download
        download_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        st.download_button(
            label="📄 Download Results (JSON)",
            data=json.dumps(download_data, indent=2),
            file_name=f"reranking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        create_visualizations(results)

def create_visualizations(results: Dict):
    """Create visualizations"""
    st.markdown("### 📈 Visualization")
    
    viz_data = []
    for technique, technique_results in results.items():
        for result in technique_results:
            viz_data.append({
                'Technique': technique,
                'Score': result['score'],
                'Rank': result['rank']
            })
    
    if viz_data:
        df = pd.DataFrame(viz_data)
        
        fig = px.bar(
            df, 
            x='Technique', 
            y='Score', 
            color='Technique',
            title="Average Scores by Technique"
        )
        st.plotly_chart(fig, use_container_width=True)

def techniques_tab():
    """Display comprehensive information about re-ranking techniques"""
    st.markdown("## 📊 Re-ranking Techniques Overview")
    
    techniques = [
        {
            "name": "Cross-Encoder",
            "description": "Joint encoding of query and document pairs for maximum accuracy",
            "pros": ["Highest accuracy", "Captures complex interactions", "State-of-the-art performance"],
            "cons": ["Computationally expensive", "Slower inference", "Requires GPU for large scale"],
            "use_case": "High-precision applications where accuracy is critical",
            "code": '''
# Cross-encoder re-ranking implementation
from sentence_transformers import CrossEncoder

def cross_encoder_rerank(query, documents, model_name='ms-marco-MiniLM-L-6-v2'):
    """
    Re-rank documents using cross-encoder architecture
    """
    # Load pre-trained cross-encoder model
    cross_encoder = CrossEncoder(model_name)
    
    # Create query-document pairs
    pairs = [(query, doc) for doc in documents]
    
    # Get relevance scores
    scores = cross_encoder.predict(pairs)
    
    # Sort by relevance score (descending)
    ranked_results = sorted(
        zip(documents, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return ranked_results
'''
        },
        {
            "name": "LLM-based Re-ranking",
            "description": "Uses large language models like GPT for intelligent document ranking",
            "pros": ["Excellent semantic understanding", "Zero-shot capability", "Handles complex queries"],
            "cons": ["API costs", "Latency issues", "Rate limiting"],
            "use_case": "Complex queries requiring deep semantic understanding",
            "code": '''
# LLM-based re-ranking using OpenAI GPT (Updated for v1.0+)
from openai import OpenAI

def llm_rerank(query, documents, api_key, top_k=10):
    """
    Use LLM to rank documents based on semantic understanding
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Prepare documents for ranking
    docs_text = "\\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    
    # Create ranking prompt
    prompt = f"""
    Query: "{query}"
    
    Documents to rank:
    {docs_text}
    
    Rank these documents by relevance to the query. 
    Return only the document numbers in order of relevance.
    Example: 3,1,5,2,4
    """
    
    # Get ranking from GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    
    # Parse and return results
    ranking_str = response.choices[0].message.content.strip()
    ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(',')]
    
    return [documents[i] for i in ranking_indices[:top_k]]
'''
        },
        {
            "name": "Cohere API",
            "description": "Production-ready re-ranking service with optimized performance",
            "pros": ["Production-ready", "Fast inference", "Multilingual support"],
            "cons": ["API dependency", "Cost per request", "Limited customization"],
            "use_case": "Production systems requiring reliable performance",
            "code": '''
# Cohere API re-ranking (Updated for v5.0+)
import cohere

def cohere_rerank(query, documents, api_key, top_k=10):
    """
    Re-rank documents using Cohere's rerank API
    """
    # Initialize Cohere client (v2)
    co = cohere.ClientV2(api_key=api_key)
    
    # Call rerank API
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_k
    )
    
    # Extract results using index to get original document text
    results = []
    for result in response.results:
        # Get original document using the index
        doc_text = documents[result.index] if result.index < len(documents) else "Unknown"
        results.append({
            'document': doc_text,
            'relevance_score': result.relevance_score,
            'index': result.index
        })
    
    return results
'''
        },
        {
            "name": "Hybrid Approach",
            "description": "Combines multiple re-ranking techniques for balanced performance",
            "pros": ["Balanced accuracy/speed", "Robust performance", "Fallback options"],
            "cons": ["Complex implementation", "Parameter tuning", "Increased maintenance"],
            "use_case": "Systems requiring both accuracy and reliability",
            "code": '''
# Hybrid re-ranking approach
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

def hybrid_rerank(query, documents, cross_encoder_model):
    """
    Combine cross-encoder with TF-IDF for hybrid ranking
    """
    # Get cross-encoder scores
    pairs = [(query, doc) for doc in documents]
{{ ... }}
    cross_scores = cross_encoder_model.predict(pairs)
    
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer()
    all_texts = [query] + documents
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Combine scores (weighted average)
    combined_scores = []
    for i, doc in enumerate(documents):
        combined_score = 0.7 * cross_scores[i] + 0.3 * tfidf_scores[i]
        combined_scores.append((doc, combined_score))
    
    return sorted(combined_scores, key=lambda x: x[1], reverse=True)
'''
        },
        {
            "name": "Learning to Rank",
            "description": "Traditional ML approach using hand-crafted features",
            "pros": ["Interpretable features", "Fast inference", "Low resource usage"],
            "cons": ["Manual feature engineering", "Limited semantic understanding", "Domain-specific"],
            "use_case": "Resource-constrained environments with clear feature requirements",
            "code": '''
# Learning to Rank with feature engineering
import numpy as np

def learning_to_rank_rerank(query, documents):
    """
    Re-rank using hand-crafted features and simple scoring
    """
    query_words = set(query.lower().split())
    scores = []
    
    for doc in documents:
        doc_words = set(doc.lower().split())
        
        # Feature extraction
        jaccard_similarity = len(query_words & doc_words) / len(query_words | doc_words)
        query_coverage = len(query_words & doc_words) / len(query_words)
        doc_length_score = min(len(doc.split()) / 20, 1.0)  # Normalize length
        
        # Weighted feature combination
        final_score = (0.5 * jaccard_similarity + 
                      0.3 * query_coverage + 
                      0.2 * doc_length_score)
        
        scores.append((doc, final_score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
'''
        }
    ]
    
    for technique in techniques:
        with st.expander(f"🔧 {technique['name']}", expanded=False):
            st.markdown(f"**Description:** {technique['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Advantages:**")
                for pro in technique['pros']:
                    st.markdown(f"• {pro}")
            
            with col2:
                st.markdown("**Limitations:**")
                for con in technique['cons']:
                    st.markdown(f"• {con}")
            
            st.markdown(f"**Best Use Case:** {technique['use_case']}")
            
            # Code example
            st.markdown("**Code Example:**")
            st.code(technique['code'], language="python")

def evaluation_tab():
    """Evaluation metrics and analysis"""
    st.markdown("## 🔬 Evaluation Metrics & Analysis")
    
    if st.session_state.reranking_results:
        st.markdown("### Current Results Analysis")
        
        results = st.session_state.reranking_results
        
        # Calculate comprehensive metrics
        metrics_data = []
        for technique, technique_results in results.items():
            if technique_results:
                scores = [r['score'] for r in technique_results]
                avg_score = np.mean(scores)
                max_score = max(scores)
                min_score = min(scores)
                std_score = np.std(scores)
                
                metrics_data.append({
                    'Technique': technique,
                    'Average Score': avg_score,
                    'Max Score': max_score,
                    'Min Score': min_score,
                    'Std Dev': std_score,
                    'Documents': len(technique_results)
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Detailed comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    df_metrics,
                    x='Technique',
                    y='Average Score',
                    title="Average Relevance Scores",
                    template="plotly_white",
                    color='Average Score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(
                    df_metrics,
                    x='Average Score',
                    y='Std Dev',
                    size='Documents',
                    color='Technique',
                    title="Score Distribution Analysis",
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Performance comparison radar chart
            if len(metrics_data) > 1:
                fig3 = go.Figure()
                
                for metric in metrics_data:
                    fig3.add_trace(go.Scatterpolar(
                        r=[metric['Average Score'], metric['Max Score'], 1-metric['Std Dev']],
                        theta=['Avg Score', 'Max Score', 'Consistency'],
                        fill='toself',
                        name=metric['Technique']
                    ))
                
                fig3.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Technique Performance Comparison"
                )
                st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("Run re-ranking in the Playground tab to see evaluation metrics here.")
    
    # Evaluation metrics explanation
    with st.expander("📖 Understanding Evaluation Metrics"):
        st.markdown("""
        ### Key Metrics for Re-ranking Evaluation
        
        **NDCG (Normalized Discounted Cumulative Gain)**
        - Measures ranking quality considering position importance
        - Higher positions get more weight in the score
        - Range: 0 to 1 (higher is better)
        - Formula: DCG@k / IDCG@k
        
        **MRR (Mean Reciprocal Rank)**
        - Measures the rank of the first relevant result
        - Focuses on finding at least one good result quickly
        - Range: 0 to 1 (higher is better)
        - Formula: 1/rank_of_first_relevant_result
        
        **Precision@K**
        - Percentage of relevant documents in top K results
        - Simple but effective metric for ranking quality
        - Range: 0 to 1 (higher is better)
        - Formula: relevant_docs_in_top_k / k
        
        **MAP (Mean Average Precision)**
        - Average of precision values at each relevant document
        - Considers both precision and recall
        - Range: 0 to 1 (higher is better)
        """)

def case_studies_tab():
    """Real-world case studies and examples"""
    st.markdown("## 📚 Real-World Case Studies")
    
    case_studies = [
        {
            "title": "E-commerce Product Search",
            "domain": "Retail",
            "challenge": "Customers searching for products often get irrelevant results due to keyword matching limitations",
            "solution": "Implemented hybrid re-ranking combining semantic similarity with user behavior signals",
            "results": ["35% improvement in click-through rate", "28% increase in conversion rate", "Reduced search abandonment by 22%"],
            "technique": "Hybrid (Cross-encoder + User signals)",
            "example": {
                "query": "wireless bluetooth headphones for running",
                "before": ["Bluetooth speaker", "Wireless mouse", "Running shoes", "Headphone stand", "Wireless headphones"],
                "after": ["Wireless sports headphones", "Bluetooth running earbuds", "Sweat-resistant wireless headphones", "Athletic wireless earphones", "Noise-canceling sports headphones"]
            }
        },
        {
            "title": "Medical Literature Search",
            "domain": "Healthcare",
            "challenge": "Doctors need to find relevant research papers quickly for evidence-based medicine",
            "solution": "Cross-encoder model fine-tuned on medical literature with domain-specific vocabulary",
            "results": ["42% improvement in finding relevant studies", "Reduced search time by 60%", "Higher physician satisfaction scores"],
            "technique": "Fine-tuned Cross-encoder",
            "example": {
                "query": "COVID-19 vaccine effectiveness elderly patients",
                "before": ["COVID-19 general information", "Vaccine development process", "Elderly care guidelines", "Patient safety protocols", "Virus transmission studies"],
                "after": ["COVID-19 vaccine efficacy in elderly populations", "Immunogenicity of COVID vaccines in older adults", "Real-world effectiveness of COVID vaccination in seniors", "Age-related vaccine response studies", "Elderly COVID vaccine safety data"]
            }
        },
        {
            "title": "Legal Document Discovery",
            "domain": "Legal",
            "challenge": "Lawyers need to find relevant case law and precedents from vast legal databases",
            "solution": "LLM-based re-ranking with legal domain expertise and citation analysis",
            "results": ["50% reduction in document review time", "Improved case preparation efficiency", "Better legal argument quality"],
            "technique": "LLM-based with domain adaptation",
            "example": {
                "query": "intellectual property patent infringement damages",
                "before": ["General patent law", "Trademark disputes", "Copyright violations", "Business litigation", "Contract law"],
                "after": ["Patent infringement damage calculations", "Intellectual property litigation precedents", "Patent damages case law", "IP infringement remedies", "Patent valuation in litigation"]
            }
        },
        {
            "title": "Customer Support Knowledge Base",
            "domain": "Customer Service",
            "challenge": "Support agents need quick access to relevant troubleshooting guides and solutions",
            "solution": "Cohere API re-ranking with real-time feedback incorporation",
            "results": ["45% faster issue resolution", "Improved first-contact resolution rate", "Higher customer satisfaction"],
            "technique": "Cohere API with feedback loops",
            "example": {
                "query": "wifi connection keeps dropping laptop",
                "before": ["General network troubleshooting", "Laptop hardware issues", "Software installation guide", "Account setup instructions", "Billing questions"],
                "after": ["WiFi connectivity troubleshooting for laptops", "Wireless adapter driver issues", "Network connection stability problems", "Laptop WiFi disconnection fixes", "Wireless network configuration guide"]
            }
        }
    ]
    
    for i, case in enumerate(case_studies):
        with st.expander(f"📋 Case Study {i+1}: {case['title']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Domain:** {case['domain']}")
                st.markdown(f"**Challenge:** {case['challenge']}")
                st.markdown(f"**Solution:** {case['solution']}")
                st.markdown(f"**Technique Used:** {case['technique']}")
                
                st.markdown("**Results Achieved:**")
                for result in case['results']:
                    st.markdown(f"• {result}")
            
            with col2:
                st.markdown("**Example Comparison:**")
                st.markdown(f"*Query:* {case['example']['query']}")
                
                st.markdown("*Before Re-ranking:*")
                for j, doc in enumerate(case['example']['before'][:3]):
                    st.markdown(f"{j+1}. {doc}")
                
                st.markdown("*After Re-ranking:*")
                for j, doc in enumerate(case['example']['after'][:3]):
                    st.markdown(f"{j+1}. {doc}")
    
    # Interactive case study demo
    st.markdown("### 🎮 Try a Case Study")
    
    selected_case = st.selectbox(
        "Select a case study to try:",
        ["Choose a case study..."] + [case['title'] for case in case_studies]
    )
    
    if selected_case != "Choose a case study...":
        case = next(case for case in case_studies if case['title'] == selected_case)
        
        if st.button(f"Load {selected_case} Example", type="secondary"):
            st.session_state.current_query = case['example']['query']
            st.session_state.current_documents = case['example']['before']
            st.success(f"Loaded {selected_case} example! Go to the Playground tab to run re-ranking.")

def about_tab():
    """About re-ranking and the Lost in the Middle problem"""
    st.markdown("## ℹ️ About Re-ranking")
    
    with st.expander("🎯 Why Re-ranking is Critical in Modern AI Systems", expanded=True):
        st.markdown("""
        ### The "Lost in the Middle" Problem
        
        Research from Stanford University ([Lost in the Middle paper](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.tacl2023.pdf)) 
        reveals a critical limitation in how Large Language Models process long contexts.
        
        **The Core Problem:**
        - LLMs perform best with information at the beginning and end of contexts
        - Information in the middle of long contexts gets "lost" or ignored
        - This significantly impacts RAG (Retrieval-Augmented Generation) systems
        - Initial retrieval often returns 100-1000 potentially relevant documents
        - Most relevant documents are scattered throughout the middle positions
        
        **Impact on Search Systems:**
        - User queries return large sets of potentially relevant documents
        - Critical information buried in middle positions gets overlooked
        - Poor document ordering leads to suboptimal LLM responses
        - User experience suffers from irrelevant or incomplete answers
        
        **Re-ranking as the Solution:**
        - **Precision Improvement**: Reassess and optimize document relevance
        - **Context Optimization**: Place most relevant documents at top positions
        - **Performance Gains**: Up to 40% improvement in search quality metrics
        - **RAG Enhancement**: Better context for LLM processing
        - **User Satisfaction**: More accurate and relevant results
        """)
    
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
        ### Two-Stage Search Architecture
        
        Modern information retrieval systems employ a sophisticated two-stage approach:
        
        **Stage 1: Retrieval (Recall-Focused)**
        - Fast filtering of large document collections (millions of documents)
        - Uses efficient algorithms like BM25, dense retrieval, or hybrid approaches
        - Returns 100-1000 candidate documents
        - Optimized for speed and broad coverage
        
        **Stage 2: Re-ranking (Precision-Focused)**
        - Precise relevance assessment of candidate documents
        - Uses complex ML models for accurate scoring
        - Returns top-k most relevant results (typically 10-50)
        - Optimized for accuracy and relevance
        
        **Benefits of This Architecture:**
        - **Efficiency**: Fast initial filtering reduces computational load
        - **Accuracy**: Sophisticated re-ranking improves result quality
        - **Scalability**: Can handle large document collections
        - **Flexibility**: Different techniques for different use cases
        """)
    
    with st.expander("🔬 Research Background"):
        st.markdown("""
        ### Key Research Findings
        
        **"Lost in the Middle" Study (Stanford, 2023)**
        - LLMs show U-shaped performance curves in long contexts
        - Information at positions 1-3 and final positions are well-processed
        - Middle positions (4-7 in a 10-document context) show significant performance drops
        - This effect is consistent across different model sizes and architectures
        
        **Implications for RAG Systems:**
        - Simply retrieving relevant documents is insufficient
        - Document ordering significantly impacts final response quality
        - Re-ranking becomes essential for optimal LLM performance
        - Strategic placement of documents can improve accuracy by 25-40%
        
        **Industry Applications:**
        - Search engines (Google, Bing)
        - Recommendation systems (Netflix, Amazon)
        - Enterprise knowledge bases
        - Customer support systems
        - Legal document analysis
        """)
    
    st.markdown("""
    ### 🚀 About This Playground
    
    This interactive playground demonstrates five different re-ranking techniques:
    
    1. **Cross-Encoder**: State-of-the-art accuracy with joint query-document encoding
    2. **LLM-based**: Leverages large language models for semantic understanding
    3. **Cohere API**: Production-ready service with optimized performance
    4. **Hybrid Approach**: Combines multiple techniques for balanced results
    5. **Learning to Rank**: Traditional ML with hand-crafted features
    
    **Features:**
    - Interactive comparison of different techniques
    - Real-time evaluation metrics and visualizations
    - Downloadable results in JSON format
    - Code examples for implementation
    - Performance analysis and insights
    """)

if __name__ == "__main__":
    main()
