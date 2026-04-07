"""
main.py - Asistente RAG - Club Social y Deportivo Las Palmeras
TP Integrador - Inteligencia Artificial - 4. anno - Lic. en Sistemas

Pipeline:
  0. Silenciar logs y warnings
  1. Configuracion hardcodeada en este archivo
  2. Cargar modelo de embeddings (all-mpnet-base-v2)
  3. Cargar/crear base vectorial Chroma
  4. Cargar LLM Qwen2.5-0.5B-Instruct
  5. Loop interactivo de preguntas por consola
"""

import os
import sys
import glob
import re
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
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline


# ============================================================
# PASO 1 - Configuracion (editar directamente aqui)
# ============================================================

# --- Rutas ---
PERSIST_DIR    = "./data_db"
DATA_DIR       = "./data"

# --- Chunking ---
CHUNK_SIZE     = 400    # Tamano de cada fragmento en caracteres
CHUNK_OVERLAP  = 50   # Solapamiento entre fragmentos consecutivos

# --- Recuperacion ---
TOP_K               = 3    # Chunks pasados al LLM: menos = mas enfocado, menos alucinacion
TOP_SHOW            = 2    # Cantidad de fuentes destacadas a mostrar al usuario
MIN_SCORE_THRESHOLD = 0.80 # Distancia maxima aceptable — solo chunks muy relevantes
                           # Si todos los chunks superan este umbral, se rechaza la consulta

# --- Modelo LLM ---
MODEL_ID         = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS   = 512
TEMPERATURE      = 0.1
REPETITION_PEN   = 1.1

# --- Modo de respuesta ---
# MODO_DIRECTO = True  -> muestra los fragmentos del documento sin pasar por el LLM
#                         Respuesta 100% fiel al texto, sin alucinaciones.
# MODO_DIRECTO = False -> usa el LLM para generar la respuesta (puede alucinar con 0.5B)
MODO_DIRECTO     = False

LINEA = "=" * 65

print(LINEA)
print("  ASISTENTE RAG - Club Social y Deportivo Las Palmeras")
print(LINEA)
print(f"  Chunk size           : {CHUNK_SIZE} chars")
print(f"  Chunk overlap        : {CHUNK_OVERLAP} chars")
print(f"  Top-K recuperado     : {TOP_K}")
print(f"  Fuentes a mostrar    : {TOP_SHOW}")
print(f"  Modelo LLM           : {MODEL_ID}")
print(f"  Temperatura          : {TEMPERATURE}")
print(f"  Repetition penalty   : {REPETITION_PEN}")
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
                loader = PyMuPDFLoader(file_path)
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
    repetition_penalty=REPETITION_PEN,
    do_sample=True,
    return_full_text=False,
)

print("[OK]  LLM listo.")


# ============================================================
# FUNCIONES DE CONSULTA RAG
# ============================================================

SYSTEM_PROMPT = (
    "Eres un asistente del Club Social y Deportivo Las Palmeras. "
    "SOLO puedes responder usando la informacion que aparece TEXTUALMENTE en el CONTEXTO proporcionado. "
    "PROHIBIDO inventar, deducir, completar o agregar cualquier dato que NO este escrito en el CONTEXTO. "
    "PROHIBIDO usar conocimiento propio o de entrenamiento. "
    "Si la respuesta no esta en el CONTEXTO, responde EXACTAMENTE: "
    "'No encuentro esa informacion en los documentos del club.' "
    "Cuando listes elementos (categorias, articulos, valores, etc.) copialos EXACTAMENTE tal como aparecen en el CONTEXTO, "
    "sin cambiar nombres, sin agregar ni quitar items."
)


def limpiar_texto(texto):
    """Normaliza texto extraido de PDF: colapsa espacios multiples y saltos de linea sueltos."""
    # Colapsar 2+ espacios en uno solo
    texto = re.sub(r'  +', ' ', texto)
    # Saltos de linea multiples con solo espacios entre ellos -> un salto de linea
    texto = re.sub(r'\n\s*\n', '\n', texto)
    return texto.strip()


def buscar_contexto(query):
    """
    Busca los TOP_K fragmentos mas relevantes en la vectorstore.
    Filtra por MIN_SCORE_THRESHOLD: descarta chunks demasiado lejanos.
    Retorna: (lista de (doc, score) unicos, texto de contexto concatenado)
    """
    k_inicial = max(TOP_K * 2, 20)
    results = vectorstore.similarity_search_with_score(query, k=k_inicial)
    results_sorted = sorted(results, key=lambda x: x[1])

    seen = set()
    unique = []
    for doc, score in results_sorted:
        if score > MIN_SCORE_THRESHOLD:
            break  # Los siguientes son aun menos relevantes; no vale la pena
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique.append((doc, score))
        if len(unique) == TOP_K:
            break

    context_text = "\n\n".join(doc.page_content for doc, _ in unique)
    return unique, context_text


def mostrar_fuentes(unique_results):
    """
    Muestra las TOP_SHOW fuentes mas relevantes con score y fragmento.
    El resto se lista de forma resumida.
    """
    print()
    print(f"  {'='*61}")
    print(f"  TOP {TOP_SHOW} FUENTES MAS RELEVANTES")
    print(f"  {'='*61}")

    destacadas = unique_results[:TOP_SHOW]
    resto      = unique_results[TOP_SHOW:]

    for i, (doc, score) in enumerate(destacadas, 1):
        fuente   = doc.metadata.get("source", "Desconocido")
        pagina   = doc.metadata.get("page", None)
        pag_str  = f" | Pag. {pagina + 1}" if pagina is not None else ""
        basename = os.path.basename(fuente)
        stars    = max(1, 5 - round(score * 2))
        estrellas = "(*) " * stars + "( ) " * (5 - stars)
        fragmento = doc.page_content.strip()[:200].replace("\n", " ")

        print(f"  #{i}  {basename}{pag_str}")
        print(f"      Score (distancia): {score:.4f}  |  Relevancia: {estrellas.strip()}")
        print(f"      Fragmento: \"{fragmento}...\"")
        print()

    if resto:
        print(f"  {'-'*61}")
        print(f"  Otras {len(resto)} fuentes consultadas (menor relevancia):")
        for i, (doc, score) in enumerate(resto, TOP_SHOW + 1):
            fuente   = doc.metadata.get("source", "Desconocido")
            pagina   = doc.metadata.get("page", None)
            pag_str  = f" p.{pagina + 1}" if pagina is not None else ""
            basename = os.path.basename(fuente)
            print(f"    #{i:>2}  dist={score:.4f}  {basename}{pag_str}")

    print(f"  {'='*61}")


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

        print()
        print(SEPARADOR)
        print('  RESPUESTA')
        print(SEPARADOR)
        print()

        if not unique_results:
            print('No encuentro evidencia suficiente en el contexto para responder esta pregunta.')
            print()
            print(f"  {'='*61}")
            print('  [!] No se hallaron fragmentos relevantes en la base vectorial.')
            print(f"      Verificá que el documento esté en 'data/' y que data_db esté actualizado.")
            print(f"  {'='*61}")
            continue

        if MODO_DIRECTO:
            # Muestra TODOS los fragmentos relevantes directamente, sin LLM
            for i, (doc, score) in enumerate(unique_results, 1):
                fuente   = doc.metadata.get('source', 'Desconocido')
                pagina   = doc.metadata.get('page', None)
                pag_str  = f' (Pag. {pagina + 1})' if pagina is not None else ''
                basename = os.path.basename(fuente)
                texto    = limpiar_texto(doc.page_content)
                print(f'  [{basename}{pag_str}]  dist={score:.4f}')
                print(f'  {"-"*61}')
                print(f'  {texto}')
                print()
        else:
            print('[...] Generando respuesta con el LLM...')
            respuesta = generar_respuesta(query, context_text)
            print(respuesta)
            print()

        mostrar_fuentes(unique_results)


if __name__ == '__main__':
    loop_interactivo()
