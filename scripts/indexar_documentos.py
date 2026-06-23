"""
Script para pré-indexar os documentos acadêmicos no Qdrant Cloud.
Execute UMA VEZ antes de iniciar o Streamlit.
O LM Studio deve ter um modelo de EMBEDDING carregado.

Uso:
    python scripts/indexar_documentos.py
"""
import os
import sys
import uuid
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTOS_DIR = os.path.join(BASE_DIR, 'documentos')
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "documentos_academicos"


def get_embed_model() -> str:
    preferidos = ["nomic", "gemma"]
    nao_queridos = ["qwen3"]
    try:
        r = requests.get(f"{LM_STUDIO_URL.rstrip('/')}/models", timeout=5)
        modelos = [m["id"] for m in r.json().get("data", []) if "embed" in m["id"].lower()]
        print(f"   Modelos de embedding disponíveis: {modelos}")
        for pref in preferidos:
            for mid in modelos:
                if pref in mid.lower() and not any(n in mid.lower() for n in nao_queridos):
                    return mid
        if modelos:
            return modelos[0]
    except Exception as e:
        print(f"   [AVISO] {e}")
    return "text-embedding-nomic-embed-text-v1.5"


def chunk_document(content: str, chunk_size: int = 400) -> list:
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 15]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def indexar():
    print("=" * 55)
    print("  INDEXAÇÃO RAG — Qdrant Cloud")
    print("=" * 55)

    if not QDRANT_URL or not QDRANT_API_KEY or "sua_url" in QDRANT_URL:
        print("\n❌ ERRO: Configure as variáveis QDRANT_URL e QDRANT_API_KEY no arquivo .env antes de continuar.")
        return

    model = get_embed_model()
    print(f"\n✅ Modelo de embedding selecionado: {model}")

    arquivos = {
        'faq_aluno.txt': 'FAQ do Aluno',
        'regulamento_academico.txt': 'Regulamento Acadêmico',
        'manual_estudos.txt': 'Manual de Estudos',
    }
    print(f"📁 Pasta: {DOCUMENTOS_DIR}")
    for f in arquivos:
        status = "✅" if os.path.exists(os.path.join(DOCUMENTOS_DIR, f)) else "❌ FALTANDO"
        print(f"   {status} {f}")

    input("\nPressione ENTER para conectar ao Qdrant e iniciar a indexação...")

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    
    # 1. Fazemos um teste de embedding para saber a dimensão
    print("\nDetectando dimensão do embedding...")
    teste_emb = client.embeddings.create(model=model, input="teste").data[0].embedding
    dimensao = len(teste_emb)
    print(f"✅ Dimensão: {dimensao}")

    # 2. Recria a coleção no Qdrant
    print("\nPreparando coleção no Qdrant Cloud...")
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
        
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dimensao, distance=Distance.COSINE),
    )
    print(f"📦 Nova coleção '{COLLECTION_NAME}' criada.")

    total_chunks = 0
    # 3. Processa e indexa os documentos
    for filename, source_name in arquivos.items():
        filepath = os.path.join(DOCUMENTOS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⏭  {source_name}: não encontrado, pulando.")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = chunk_document(content)
        print(f"\n📄 {source_name}: {len(chunks)} chunks")

        points = []
        for i, chunk in enumerate(chunks):
            sys.stdout.write(f"\r   [{i+1}/{len(chunks)}] Vetorizando...")
            sys.stdout.flush()

            try:
                resp = client.embeddings.create(model=model, input=chunk, timeout=30)
                embedding = resp.data[0].embedding

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": source_name,
                            "filename": filename
                        }
                    )
                )
            except Exception as e:
                print(f"\n   ❌ Erro no chunk {i+1}: {e}")
                print("      Pulando e continuando...")

        if points:
            sys.stdout.write(f"\r   Enviando {len(points)} chunks para o Qdrant Cloud...")
            sys.stdout.flush()
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            total_chunks += len(points)
            print(f"\r   ✅ {source_name} concluído: {len(points)} chunks indexados.     ")

    print(f"\n{'='*55}")
    print(f"✅ INDEXAÇÃO CONCLUÍDA!")
    print(f"   {total_chunks} chunks salvos na nuvem do Qdrant!")
    print(f"{'='*55}")
    print("\n➡  Agora você pode iniciar o Streamlit: streamlit run app.py --server.port 8502\n")


if __name__ == "__main__":
    try:
        indexar()
    except Exception as e:
        import traceback
        print(f"\n❌ ERRO FATAL: {e}")
        traceback.print_exc()
