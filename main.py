"""
main.py - Asistente RAG - Club Social y Deportivo Las Palmeras
TP Integrador - Inteligencia Artificial - 4. anno - Lic. en Sistemas

Pipeline:
  0. Silenciar logs y warnings
  1. Leer config.json
  2. Cargar modelo de embeddings (all-mpnet-base-v2)
  3. Cargar/crear base vectorial Chroma
  4. Cargar LLM Qwen2.5-0.5B-Instruct
  5. Loop interactivo de preguntas por consola
"""

import os
import sys
import glob
import json
import warnings
import logging

# ============================================================
# PASO 0 - Silenciar logs, warnings y telemetria
# ============================================================
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

_SILENT = logging.CRITICAL + 1

logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

for _noisy in (
    "pypdf", "pypdf._cmap", "pypdf._page",
    "huggingface_hub", "huggingface_hub.utils._http",
    "transformers", "sentence_transformers",
    "langchain", "langchain_core", "chromadb",
):
    logging.getLogger(_noisy).setLevel(_SILENT)

# ------------------------------------------------------------
# Importaciones principales (despues de silenciar)
# ------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline


# ============================================================
# PASO 1 - Leer configuracion desde config.json
# ============================================================

CONFIG_PATH = "config.json"

CONFIG_PRESETS = {
    "A": {"chunk_size": 300,  "chunk_overlap": 50,  "top_k": 5},
    "B": {"chunk_size": 800,  "chunk_overlap": 150, "top_k": 10},
    "C": {"chunk_size": 500,  "chunk_overlap": 100, "top_k": 5},
}

DEFAULTS = {
    "config_name": "B",
    "chunk_size": 800,
    "chunk_overlap": 150,
    "top_k": 10,
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.7,
}


def cargar_config(path):
    """
    Carga config.json. Si no existe, usa DEFAULTS (Config B).
    Si config_name coincide con un preset (A/B/C), los parametros
    del preset sobreescriben chunk_size, chunk_overlap y top_k.
    """
    if not os.path.exists(path):
        print("[AVISO] No se encontro config.json. Usando configuracion B por defecto.")
        return dict(DEFAULTS)

    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    for key, val in DEFAULTS.items():
        cfg.setdefault(key, val)

    preset_name = str(cfg.get("config_name", "")).upper()
    if preset_name in CONFIG_PRESETS:
        for key, val in CONFIG_PRESETS[preset_name].items():
            cfg[key] = val
        cfg["config_name"] = preset_name

    return cfg


cfg = cargar_config(CONFIG_PATH)

PERSIST_DIR    = "./data_db"
DATA_DIR       = "./data"
CHUNK_SIZE     = cfg["chunk_size"]
CHUNK_OVERLAP  = cfg["chunk_overlap"]
TOP_K          = cfg["top_k"]
MODEL_ID       = cfg["model_id"]
MAX_NEW_TOKENS = cfg["max_new_tokens"]
TEMPERATURE    = cfg["temperature"]
CONFIG_NAME    = cfg["config_name"]

LINEA = "=" * 65

print(LINEA)
print("  ASISTENTE RAG - Club Social y Deportivo Las Palmeras")
print(LINEA)
print(f"  Configuracion activa : Config {CONFIG_NAME}")
print(f"  Chunk size           : {CHUNK_SIZE} chars")
print(f"  Chunk overlap        : {CHUNK_OVERLAP} chars")
print(f"  Top-K recuperado     : {TOP_K}")
print(f"  Modelo LLM           : {MODEL_ID}")
print(f"  Temperature          : {TEMPERATURE}")
print(LINEA)


# ============================================================
# PASO 2 - Cargar modelo de embeddings
# ============================================================

print("\nCargando modelo de embeddings (all-mpnet-base-v2)...")

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

print("[OK] Embeddings listos (768 dims)")


# ============================================================
# PASOS 3-4 - Cargar o construir la base vectorial
# ============================================================

def cargar_documentos(data_dir):
    """Recorre data_dir recursivamente y carga todos los .txt / .pdf / .csv."""
    raw_docs = []
    pattern = os.path.join(data_dir, "**", "*")
    for file_path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(file_path):
            continue
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                print(f"  [SKIP] Formato no soportado: {os.path.basename(file_path)}")
                continue

            # Suprimir salida de librerias ruidosas durante la carga
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
            try:
                docs = loader.load()
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr

            raw_docs.extend(docs)
            rel = os.path.relpath(file_path, data_dir)
            print(f"  [OK] {len(docs):>3} doc(s)  <--  {rel}")

        except Exception as exc:
            print(f"  [ERROR] {os.path.basename(file_path)}: {exc}")

    return raw_docs


if os.path.exists(PERSIST_DIR):
    print()
    print(f"[INFO] Base vectorial encontrada en '{PERSIST_DIR}'.")
    print("[INFO] Cargando DB existente (se omite re-indexacion)...")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    print("[OK]  Base vectorial cargada correctamente.")

else:
    print()
    print("[AVISO] No se encontro data_db/. Construyendo base vectorial desde cero...")

    print(f"\n[PASO 3a] Cargando documentos desde '{DATA_DIR}'...")
    raw_documents = cargar_documentos(DATA_DIR)
    print(f"[INFO] Total de documentos cargados: {len(raw_documents)}")

    if not raw_documents:
        print("[ERROR] No se encontraron documentos en la carpeta 'data/'. Abortando.")
        sys.exit(1)

    print("\n[PASO 3b] Dividiendo en fragmentos (chunks)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    documents = splitter.split_documents(raw_documents)
    print(f"[INFO] Chunks generados : {len(documents)}")
    print(f"[INFO] Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")

    print(f"\n[PASO 4] Creando base vectorial en '{PERSIST_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print("[OK]  Base vectorial creada y persistida.")
    print(f"[OK]  Total de chunks indexados: {len(documents)}")


# ============================================================
# PASO 5 - Cargar el LLM Qwen2.5-0.5B-Instruct
# ============================================================

print(f"\n[PASO 5] Cargando LLM ({MODEL_ID})...")
print("         Esto puede tardar varios minutos la primera vez.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)

pipe = hf_pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    return_full_text=False,
)

print("[OK]  LLM listo.")


# ============================================================
# FUNCIONES DE CONSULTA RAG
# ============================================================

SYSTEM_PROMPT = (
    "Responde UNICAMENTE con informacion presente en el contexto proporcionado. "
    "Si la respuesta no esta en el contexto, di exactamente: "
    "No encuentro evidencia suficiente en el contexto para responder esta pregunta. "
    "No uses conocimiento general ni suposiciones."
)


def buscar_contexto(query):
    """
    Busca los TOP_K fragmentos mas relevantes en la vectorstore.
    Retorna: (lista de (doc, score) unicos, texto de contexto concatenado)
    """
    k_inicial = max(TOP_K * 2, 20)
    results = vectorstore.similarity_search_with_score(query, k=k_inicial)
    results_sorted = sorted(results, key=lambda x: x[1])

    seen = set()
    unique = []
    for doc, score in results_sorted:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique.append((doc, score))
        if len(unique) == TOP_K:
            break

    context_text = "\n\n".join(doc.page_content for doc, _ in unique)
    return unique, context_text


def mostrar_fuentes(unique_results):
    """Imprime las fuentes de los fragmentos recuperados (trazabilidad)."""
    print()
    print(f"  {'-'*61}")
    print(f"  FUENTES CONSULTADAS ({len(unique_results)} fragmentos recuperados)")
    print(f"  {'-'*61}")
    for i, (doc, score) in enumerate(unique_results, 1):
        fuente  = doc.metadata.get("source", "Desconocido")
        pagina  = doc.metadata.get("page", None)
        pag_str = f" | Pag. {pagina + 1}" if pagina is not None else ""
        stars   = max(1, 5 - round(score * 2))
        estrellas = "(*)" * stars + "( )" * (5 - stars)
        basename = os.path.basename(fuente)
        print(f"  #{i:>2}  [{estrellas}]  dist={score:.4f}  {basename}{pag_str}")
    print(f"  {'-'*61}")


def generar_respuesta(query, context_text):
    """Genera respuesta con el LLM a partir del contexto recuperado."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "CONTEXTO:\n" + context_text + "\n\nPREGUNTA:\n" + query},
    ]
    final_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = pipe(final_prompt)
    response = output[0]["generated_text"].strip()


    tokens_to_clean = [
        '<|im_end|>',
        '<|endoftext|>',
    ]
    for tok in tokens_to_clean:
        response = response.replace(tok, '')
    return response.strip()


SEPARADOR = '=' * 65


def loop_interactivo():
    print()
    print(SEPARADOR)
    print('  SISTEMA LISTO. MODO INTERACTIVO')
    print('  Escribi tu pregunta y presiona ENTER.')
    print('  Escribe  salir  para terminar.')
    print(SEPARADOR)

    while True:
        print()
        try:
            query = input('  Pregunta: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print('[INFO] Sesion terminada por el usuario.')
            break

        if not query:
            print('[AVISO] Pregunta vacia. Intenta de nuevo.')
            continue

        if query.lower() in ('salir', 'exit', 'quit', 'q'):
            print('[INFO] Hasta luego!')
            break

        print()
        print('[...] Buscando fragmentos relevantes en la base vectorial...')
        unique_results, context_text = buscar_contexto(query)

        print('[...] Generando respuesta con el LLM...')
        respuesta = generar_respuesta(query, context_text)

        print()
        print(SEPARADOR)
        print('  RESPUESTA')
        print(SEPARADOR)
        print()
        print(respuesta)
        print()

        mostrar_fuentes(unique_results)


if __name__ == '__main__':
    loop_interactivo()
