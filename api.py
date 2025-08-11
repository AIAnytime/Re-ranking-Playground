from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from openai import OpenAI
import cohere
import numpy as np
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = FastAPI(title="Re-ranking API", description="Production-grade re-ranking service")

# Request/Response models
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    technique: str = "cross_encoder"
    top_k: int = 10

class RerankResponse(BaseModel):
    query: str
    technique: str
    results: List[Dict]
    total_documents: int
    processing_time: float

# Global models
models = {}

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global models
    try:
        # Cross-encoder model - try different model names
        try:
            models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder model loaded successfully")
        except:
            try:
                models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
                print("✅ Cross-encoder (TinyBERT) model loaded successfully")
            except Exception as e:
                print(f"⚠️ Cross-encoder model not available: {e}")
        
        # Cohere client
        cohere_key = os.getenv('COHERE_API_KEY')
        if cohere_key and cohere_key.strip():
            try:
                models['cohere'] = cohere.ClientV2(api_key=cohere_key)
                print("✅ Cohere API client loaded successfully")
            except Exception as e:
                print(f"⚠️ Cohere API setup failed: {e}")
        else:
            print("⚠️ Cohere API key not found")
        
        # OpenAI setup
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key.strip():
            try:
                models['openai'] = OpenAI(api_key=openai_key)
                print("✅ OpenAI API client loaded successfully")
            except Exception as e:
                print(f"⚠️ OpenAI API setup failed: {e}")
        else:
            print("⚠️ OpenAI API key not found")
            
        print("🚀 API service ready")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

class ProductionRerankingService:
    """Production-grade re-ranking service"""
    
    @staticmethod
    def cross_encoder_rerank(query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Cross-encoder re-ranking with optimized batch processing"""
        try:
            model = models.get('cross_encoder')
            if not model:
                raise HTTPException(status_code=500, detail="Cross-encoder model not available")
            
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get relevance scores
            scores = model.predict(pairs)
            
            # Sort by relevance score
            ranked_results = sorted(
                zip(documents, scores), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            # Format results
            results = [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1
                }
                for i, (doc, score) in enumerate(ranked_results)
            ]
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cross-encoder error: {str(e)}")
    
    @staticmethod
    def cohere_rerank(query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Cohere API re-ranking"""
        try:
            client = models.get('cohere')
            if not client:
                raise HTTPException(status_code=500, detail="Cohere API not available")
            
            # Call Cohere rerank API with new v2 client
            response = client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            # Format results using the index to get original document text
            results = []
            for i, result in enumerate(response.results):
                # Get the original document text using the index
                doc_index = result.index
                doc_text = documents[doc_index] if doc_index < len(documents) else "Unknown document"
                
                results.append({
                    'document': doc_text,
                    'score': result.relevance_score,
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cohere error: {str(e)}")
    
    @staticmethod
    def llm_rerank(query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """LLM-based re-ranking using OpenAI"""
        try:
            openai_client = models.get('openai')
            if not openai_client:
                raise HTTPException(status_code=500, detail="OpenAI API not available")
            
            # Prepare prompt
            docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
            
            prompt = f"""
            Query: "{query}"
            
            Documents to rank:
            {docs_text}
            
            Rank these documents by relevance to the query. Return only the document numbers in order of relevance (most relevant first), separated by commas.
            Example response: 3,1,5,2,4
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            
            # Parse ranking
            ranking_str = response.choices[0].message.content.strip()
            ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
            
            # Create results
            results = []
            for i, doc_idx in enumerate(ranking_indices[:top_k]):
                if 0 <= doc_idx < len(documents):
                    results.append({
                        'document': documents[doc_idx],
                        'score': 1.0 - (i * 0.05),  # Simulated decreasing score
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    @staticmethod
    def hybrid_rerank(query: str, documents: List[str], top_k: int = 10) -> List[Dict]:
        """Hybrid re-ranking combining multiple approaches"""
        try:
            # Get cross-encoder scores
            cross_results = ProductionRerankingService.cross_encoder_rerank(query, documents, len(documents))
            
            # Simple TF-IDF similarity as second signal
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer()
            all_texts = [query] + documents
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Combine scores (70% cross-encoder, 30% TF-IDF)
            combined_scores = []
            for i, doc in enumerate(documents):
                cross_score = cross_results[i]['score'] if i < len(cross_results) else 0
                tfidf_score = similarities[i]
                combined_score = 0.7 * cross_score + 0.3 * tfidf_score
                combined_scores.append((doc, combined_score))
            
            # Sort and format
            sorted_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
            
            results = [
                {
                    'document': doc,
                    'score': float(score),
                    'rank': i + 1
                }
                for i, (doc, score) in enumerate(sorted_results)
            ]
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Hybrid error: {str(e)}")

# API endpoints
@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Re-rank documents using specified technique"""
    import time
    start_time = time.time()
    
    try:
        # Validate input
        if not request.query or not request.documents:
            raise HTTPException(status_code=400, detail="Query and documents are required")
        
        # Route to appropriate re-ranking technique
        service = ProductionRerankingService()
        
        if request.technique == "cross_encoder":
            results = service.cross_encoder_rerank(request.query, request.documents, request.top_k)
        elif request.technique == "cohere":
            results = service.cohere_rerank(request.query, request.documents, request.top_k)
        elif request.technique == "llm":
            results = service.llm_rerank(request.query, request.documents, request.top_k)
        elif request.technique == "hybrid":
            results = service.hybrid_rerank(request.query, request.documents, request.top_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid technique specified")
        
        processing_time = time.time() - start_time
        
        return RerankResponse(
            query=request.query,
            technique=request.technique,
            results=results,
            total_documents=len(request.documents),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Re-ranking API is running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
