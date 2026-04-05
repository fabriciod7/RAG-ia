# Asistente RAG — Club Social y Deportivo
**TP Integrador · Inteligencia Artificial · 4.º año · Licenciatura en Sistemas**

---

## Descripción del proyecto

Sistema RAG (Retrieval-Augmented Generation) que responde preguntas sobre el reglamento, procedimientos y servicios de un club social y deportivo. Toda la información proviene exclusivamente del corpus documental interno del club. El sistema está diseñado para rechazar preguntas cuando no existe evidencia suficiente en los documentos cargados.

El pipeline corre **completamente en local**: no requiere API keys ni conexión a servicios externos.

---

## Estructura del proyecto

```
rag_club/
├── main.py                  # Punto de entrada: demo interactiva y comparación de configs
├── requirements.txt         # Dependencias del proyecto
├── .gitignore
│
├── data/                    # Corpus documental (fuentes primarias)
│   ├── reglamentos/         # Estatutos, reglamento de socios, reglamento de torneos
│   ├── procedimientos/      # Alta de socios, pagos, reserva de instalaciones
│   └── faq/                 # Preguntas frecuentes reales recopiladas del club
│
└── data_db/                 # Base vectorial Chroma (generada automáticamente, no editar)
```

> `data_db/` se genera sola la primera vez que se ejecuta `main.py`. Si se modifica el corpus, eliminar esta carpeta para forzar la re-indexación.

---

## Stack tecnológico

| Componente | Tecnología | Detalle |
|---|---|---|
| Carga de documentos | `langchain-community` | `PyPDFLoader`, `TextLoader`, `CSVLoader` |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| Embeddings | `langchain-huggingface` | `HuggingFaceEmbeddings` · modelo `all-mpnet-base-v2` |
| Base vectorial | `langchain-chroma` | `Chroma` con persistencia en `data_db/` |
| LLM | `transformers` + `langchain-huggingface` | `Qwen/Qwen2.5-0.5B-Instruct` · inferencia local |
| Aceleración | `torch` | `device_map="auto"` · usa GPU si está disponible |

Todo el stack corre en local. No se requiere ninguna API key.

---

## Requisitos del entorno

### Python
Versión recomendada: **Python 3.10+**

### Entorno virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Contenido de `requirements.txt`

```
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-chroma
langchain-text-splitters
chromadb
sentence-transformers
transformers
torch
accelerate
pypdf
```

> **Nota sobre PyTorch:** si tu máquina tiene GPU NVIDIA, reemplazá la línea `torch` por la versión con soporte CUDA. Ver: https://pytorch.org/get-started/locally/  
> Sin GPU, el modelo corre en CPU con mayor latencia pero sin errores.

---

## Corpus documental (`data/`)

Todos los documentos son archivos `.txt`, `.pdf` o `.csv` redactados por el grupo a partir de información relevada del club. El loader detecta el formato automáticamente.

### `data/reglamentos/`
| Archivo | Contenido |
|---|---|
| `reglamento_socios.txt` | Categorías de socios (activo, adherente, cadete, vitalicio), cuotas por categoría, beneficios, causas de baja o suspensión |
| `reglamento_torneos.txt` | Categorías por edad, sistema de puntos, criterios de desempate, sanciones |

### `data/procedimientos/`
| Archivo | Contenido |
|---|---|
| `alta_socios.txt` | Pasos para inscribirse, documentación requerida, plazos |
| `pagos_y_mora.txt` | Medios de pago, fechas de vencimiento, consecuencias de la mora, cómo regularizar |
| `reserva_instalaciones.txt` | Quién puede reservar, con cuánta anticipación, política de cancelación, costos adicionales |

### `data/faq/`
| Archivo | Contenido |
|---|---|
| `preguntas_frecuentes.txt` | 15 preguntas reales que el personal del club recibe habitualmente, con sus respuestas |

> **Trazabilidad del corpus:** al incorporar cada documento, registrar en la bitácora: origen (entrevista / documento interno / transcripción), fecha de recopilación y motivo de inclusión. Requisito explícito de la consigna.

---

## Pipeline técnico

```
data/  (.txt / .pdf / .csv)
    │
    ▼
[1] Carga          PyPDFLoader / TextLoader / CSVLoader
    │               (detección automática por extensión)
    ▼
[2] Chunking       RecursiveCharacterTextSplitter
    │               chunk_size=800 · overlap=150 (config base)
    ▼
[3] Embeddings     HuggingFaceEmbeddings
    │               modelo: all-mpnet-base-v2 (768 dims)
    ▼
[4] Vectorstore    Chroma
    │               persistido en data_db/
    │               se reutiliza si ya existe
    ▼
[5] Consulta       query del usuario
    │
    ▼
[6] Búsqueda       similarity_search_with_score
    │               k=20 → deduplicación → top-k seleccionado
    ▼
[7] Prompt         Contexto + instrucción de rechazo + pregunta
    │
    ▼
[8] LLM            Qwen2.5-0.5B-Instruct (local)
    │               max_new_tokens=512 · temperature=0.7
    ▼
    Respuesta + fuentes (trazabilidad)
```

---

## Configuraciones comparadas

La consigna exige probar al menos 3 variantes. Se propone variar los parámetros del chunking y el top-k:

| Config | Chunk size | Overlap | Top-k recuperado | Diferencia esperada |
|--------|-----------|---------|-----------------|---------------------|
| **A** | 300 chars | 50 | 5 | Fragmentos pequeños y precisos — mejor en preguntas directas |
| **B** | 800 chars | 150 | 10 | Más contexto por fragmento — mejor en preguntas de integración |
| **C** | 500 chars | 100 | 5 | Balance intermedio — punto de comparación neutral |

Para comparar, correr `main.py` tres veces cambiando los parámetros y registrar los resultados en la bitácora. Eliminar `data_db/` entre cada corrida para asegurar que los chunks no se mezclen.

---

## Ejecución

### Primera ejecución (construye la base vectorial)

```bash
python main.py
```

El script detecta si `data_db/` existe:
- **No existe** → carga documentos, genera chunks, calcula embeddings, persiste Chroma.
- **Ya existe** → carga directamente y va a la consulta. Más rápido.

### Cambiar la consulta

Editar esta línea en `main.py`:

```python
query = "¿Cuánto paga de cuota un socio adherente?"
```

### Regenerar la base vectorial

Si se agregan o modifican documentos del corpus:

```bash
rm -rf data_db/       # Linux / macOS
rmdir /s data_db      # Windows
python main.py
```

---

## Prompt y comportamiento de rechazo

El system prompt del modelo debe instruir explícitamente el rechazo cuando no haya evidencia:

```python
{
    "role": "system",
    "content": (
        "Respondé ÚNICAMENTE con información presente en el contexto proporcionado. "
        "Si la respuesta no está en el contexto, decí exactamente: "
        "'No encuentro evidencia suficiente en el contexto para responder esta pregunta.' "
        "No uses conocimiento general ni suposiciones."
    )
}
```

Este comportamiento se verifica manualmente con las preguntas de borde definidas antes de ajustar el sistema.

---

## Decisiones técnicas a documentar en la bitácora

Durante el desarrollo, registrar al menos:

- Por qué se eligió `all-mpnet-base-v2` y no otro modelo de embeddings
- Por qué `temperature=0.7` en el LLM y qué pasaría con `temperature=0`
- Qué separadores usa `RecursiveCharacterTextSplitter` y en qué orden los aplica
- Cómo se definió la instrucción de rechazo en el prompt y qué alternativas se probaron
- Qué falló en la primera versión y cómo se corrigió

---

## `.gitignore` recomendado

```
data_db/
.venv/
__pycache__/
*.pyc
*.pth
```

---

## Uso declarado de IA (según consigna)

Completar en el informe final:

| Herramienta | Para qué se usó | Qué validó o modificó el grupo |
|---|---|---|
| | | |
