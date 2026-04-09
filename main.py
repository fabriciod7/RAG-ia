import os
import sys
import glob
import re
import json
import warnings
import logging
import time
from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe
load_dotenv()

# PASO 0 - Silenciar logs, warnings y telemetria
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

_SILENT = logging.CRITICAL + 1

logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

for _noisy in (
    "chromadb", "langchain", "langchain_core", "google.generativeai"
):
    logging.getLogger(_noisy).setLevel(_SILENT)

# Importaciones principales
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PASO 1 - Configuracion (editar directamente aqui)
# Rutas 
PERSIST_DIR    = "./data_db"
DATA_DIR       = "./data"
METADATA_FILE  = os.path.join(DATA_DIR, "metadata.json")

# Chunking
CHUNK_SIZE     = 400    # Tamano de cada fragmento en caracteres
CHUNK_OVERLAP  = 50   # Solapamiento entre fragmentos consecutivos

# Recuperacion
TOP_K               = 12   # Los K mejores
TOP_SHOW            = 3    # Muestra 3 fuentes relevantes
MIN_SCORE_THRESHOLD = 2.00 # Distancia maxima aceptable.

# Cargar API KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_ID         = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL  = "gemini-embedding-2-preview"
MAX_NEW_TOKENS   = 1024
TEMPERATURE      = 0.1

# Modo de respuesta
# MODO_DIRECTO = True  -> muestra los fragmentos del documento sin pasar por el LLM, Respuesta 100% fiel al texto, sin alucinaciones.
# MODO_DIRECTO = False -> usa el LLM para generar la respuesta
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
print(f"  Modelo Embeddings    : {EMBEDDING_MODEL}")
print(f"  Temperatura          : {TEMPERATURE}")
print(LINEA)

if not GOOGLE_API_KEY:
    print("\n[!] ERROR: No se encontro la variable de entorno 'GOOGLE_API_KEY'.")
    print("    Por favor, obten una en https://aistudio.google.com/app/apikey")
    print("    Y configurala en un archivo .env asi: GOOGLE_API_KEY=tu_clave_aca")
    print("\n[INFO] El programa se cerrara ahora.")
    sys.exit(1)

# PASO 2 - Cargar modelo de embeddings (Gemini)

print(f"\nCargando embeddings ({EMBEDDING_MODEL})...")

# Limite de peticiones por minuto
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1.0, 
    max_bucket_size=10
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
    rate_limiter=rate_limiter
)
print("[OK] Embeddings configurados (con limitador de tasa)")

# PASOS 3-4 - Cargar o construir la base vectorial
def cargar_metadata(metadata_file):
    """Carga metadata.json y devuelve un dict {archivo_relativo: info_doc}."""
    if not os.path.exists(metadata_file):
        print("[AVISO] No se encontro metadata.json. Se cargaran documentos sin metadata adicional.")
        return {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for doc_info in data.get("documentos", []):
        archivo = doc_info.get("archivo", "").replace("/", os.sep)
        lookup[archivo] = doc_info
    print(f"[OK] Metadata cargada: {len(lookup)} documentos registrados.")
    return lookup

def cargar_documentos(data_dir, metadata_lookup=None):
    """Recorre data_dir recursivamente y carga todos los .txt / .pdf / .csv.
    Enriquece cada documento con metadata del JSON si esta disponible."""
    if metadata_lookup is None:
        metadata_lookup = {}
    raw_docs = []
    pattern = os.path.join(data_dir, "**", "*")
    for file_path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(file_path):
            continue
        # Saltar archivos de metadata
        if os.path.basename(file_path) == "metadata.json":
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

            # Enriquecer documentos con metadata del JSON
            rel = os.path.relpath(file_path, data_dir)
            meta_info = metadata_lookup.get(rel, None)
            if meta_info:
                for doc in docs:
                    doc.metadata["titulo"] = meta_info.get("titulo", "")
                    doc.metadata["categoria"] = meta_info.get("categoria", "")
                    doc.metadata["prioridad"] = meta_info.get("prioridad", 99)
                    doc.metadata["doc_id"] = meta_info.get("id", "")
                tag = f" [P{meta_info['prioridad']}]"
            else:
                for doc in docs:
                    doc.metadata["prioridad"] = 99
                tag = ""

            raw_docs.extend(docs)
            print(f"  [OK] {len(docs):>3} doc(s)  <--  {rel}{tag}")

        except Exception as exc:
            print(f"  [ERROR] {os.path.basename(file_path)}: {exc}")

    return raw_docs


if os.path.exists(PERSIST_DIR):
    print()
    print(f"[INFO] Base vectorial encontrada en '{PERSIST_DIR}'.")
    
    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        # Una pequeña prueba para ver si las dimensiones coinciden
        vectorstore.similarity_search("test", k=1)
        print("[OK]  Base vectorial cargada y validada.")
    except Exception as e:
        print(f"[AVISO] Error al cargar la base existente: {e}")
        print("[AVISO] Es probable que el modelo de embeddings haya cambiado.")
        print("[AVISO] SE ELIMINARA LA BASE ACTUAL Y SE RE-INDEXARA...")
        import shutil
        shutil.rmtree(PERSIST_DIR)
        raw_documents = None # Forzar carga en el else

else:
    print()
    print("[AVISO] No se encontro data_db/. Construyendo base vectorial desde cero...")

    print(f"\n[PASO 3a] Cargando documentos desde '{DATA_DIR}'...")
    metadata_lookup = cargar_metadata(METADATA_FILE)
    raw_documents = cargar_documentos(DATA_DIR, metadata_lookup)
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
    print(f"[INFO] Indexando {len(documents)} fragmentos en lotes para respetar limites de la API...")

    # Inicializar Chroma vacio
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Indexar en lotes pequenos por si la API procesa 1 request por documento
    # Con 10 docs cada 7 segundos, estamos en ~85 peticiones por minuto (tope Free Tier = 100)
    BATCH_SIZE = 10
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"       -> procesando fragmentos {i+1} a {min(i + BATCH_SIZE, len(documents))}...")
        vectorstore.add_documents(batch)
        if i + BATCH_SIZE < len(documents):
            time.sleep(7) # Espera segura para Free Tier

    print("[OK]  Base vectorial creada y persistida.")
    print(f"[OK]  Total de chunks indexados: {len(documents)}")

# PASO 5 - Cargar el LLM Gemini 1.5 Flash
print(f"\n[PASO 5] Conectando con {MODEL_ID}...")

llm = ChatGoogleGenerativeAI(
    model=MODEL_ID,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
    max_output_tokens=MAX_NEW_TOKENS,
)
print("[OK]  LLM listo (via API).")

# FUNCIONES DE CONSULTA RAG
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
    # Ordenar por distancia; ante scores similares (±0.01) priorizar segun metadata
    results_sorted = sorted(results, key=lambda x: (round(x[1], 2), x[0].metadata.get("prioridad", 99)))

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
        fuente    = doc.metadata.get("source", "Desconocido")
        pagina    = doc.metadata.get("page", None)
        titulo    = doc.metadata.get("titulo", "")
        categoria = doc.metadata.get("categoria", "")
        pag_str   = f" | Pag. {pagina + 1}" if pagina is not None else ""
        basename  = os.path.basename(fuente)
        stars     = max(1, 5 - round(score * 2))
        estrellas = "(*) " * stars + "( ) " * (5 - stars)
        fragmento = doc.page_content.strip()[:200].replace("\n", " ")

        nombre_doc = f"{titulo} ({basename})" if titulo else basename
        meta_str   = f"  [{categoria.upper()}]" if categoria else ""

        print(f"  #{i}  {nombre_doc}{pag_str}{meta_str}")
        print(f"      Score (distancia): {score:.4f}  |  Relevancia: {estrellas.strip()}")
        print(f"      Fragmento: \"{fragmento}...\"")
        print()

    if resto:
        print(f"  {'-'*61}")
        print(f"  Otras {len(resto)} fuentes consultadas (menor relevancia):")
        for i, (doc, score) in enumerate(resto, TOP_SHOW + 1):
            fuente   = doc.metadata.get("source", "Desconocido")
            pagina   = doc.metadata.get("page", None)
            titulo   = doc.metadata.get("titulo", "")
            pag_str  = f" p.{pagina + 1}" if pagina is not None else ""
            basename = os.path.basename(fuente)
            nombre_doc = f"{titulo} ({basename})" if titulo else basename
            print(f"    #{i:>2}  dist={score:.4f}  {nombre_doc}{pag_str}")

    print(f"  {'='*61}")

def generar_respuesta(query, context_text):
    """Genera respuesta con Gemini a partir del contexto recuperado."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXTO:\n{context_text}\n\nPREGUNTA:\n{query}"),
    ]
    
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        # Langchain puede devolver una lista de bloques con Gemini
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()
    
    return str(content).strip()

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
                titulo   = doc.metadata.get('titulo', '')
                pag_str  = f' (Pag. {pagina + 1})' if pagina is not None else ''
                basename = os.path.basename(fuente)
                nombre_doc = f'{titulo} ({basename})' if titulo else basename
                texto    = limpiar_texto(doc.page_content)
                print(f'  [{nombre_doc}{pag_str}]  dist={score:.4f}')
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