"""
RAG (Retrieval-Augmented Generation) para documentos acadêmicos.
Usa Qdrant Cloud como vector store e embeddings do LM Studio.
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "documentos_academicos"

_client_cache = None
_model_cache = None
_qdrant_cache = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_cache
    if _qdrant_cache is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("As variáveis QDRANT_URL e QDRANT_API_KEY não estão definidas no .env")
        _qdrant_cache = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
    return _qdrant_cache


def _get_openai() -> OpenAI:
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _client_cache


def _get_embed_model() -> str:
    global _model_cache
    if _model_cache:
        return _model_cache
    preferidos = ["nomic", "gemma"]
    nao_queridos = ["qwen3"]
    try:
        r = requests.get(f"{LM_STUDIO_URL.rstrip('/')}/models", timeout=5)
        modelos = [m["id"] for m in r.json().get("data", []) if "embed" in m["id"].lower()]
        for pref in preferidos:
            for mid in modelos:
                if pref in mid.lower() and not any(n in mid.lower() for n in nao_queridos):
                    _model_cache = mid
                    return _model_cache
        if modelos:
            _model_cache = modelos[0]
            return _model_cache
    except Exception:
        pass
    _model_cache = "text-embedding-nomic-embed-text-v1.5"
    return _model_cache


def _embed(text: str) -> list:
    """Gera embedding para um texto via LM Studio."""
    client = _get_openai()
    model = _get_embed_model()
    resp = client.embeddings.create(model=model, input=text, timeout=30)
    return resp.data[0].embedding


def buscar_documentos_rag(query: str, n_results: int = 4) -> str:
    """
    Busca semanticamente os trechos mais relevantes dos documentos acadêmicos no Qdrant Cloud.
    """
    try:
        qdrant = _get_qdrant()
        
        # Verifica se a coleção existe
        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            return "⚠️ A coleção no Qdrant não existe. Execute o script de indexação primeiro."

        # Vetoriza a query
        query_vec = _embed(query)

        # Busca no Qdrant
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=n_results
        )

        if not search_result.points:
            return "Nenhum trecho relevante encontrado para esta pergunta."

        output = "=== DOCUMENTOS ACADÊMICOS RELEVANTES (RAG) ===\n\n"
        for hit in search_result.points:
            relevancia = round(hit.score * 100, 1)
            payload = hit.payload
            output += f"[Fonte: {payload['source']} | Relevância: {relevancia}%]\n"
            output += f"{payload['text']}\n\n"

        return output

    except Exception as e:
        import traceback
        return f"Erro na busca RAG no Qdrant: {e}\n{traceback.format_exc()}"
