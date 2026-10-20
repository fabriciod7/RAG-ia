# Sistema RAG orientado a la reglamentacion y normativas de un club deportivo.

GUIA DE EJECUCION

# Requisitos Previos

- Tener instalado **Python 3.10** o superior.
- Una **API Key de Google Gemini** (gratuita). Puedes obtenerla en: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

## Paso 1: Entorno Virtual
Abre tu terminal (Símbolo de sistema o PowerShell) en la carpeta raíz del proyecto y crea un entorno virtual para aislar las dependencias:
```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows:
.venv\Scripts\activate

# Activar en Linux/macOS:
source .venv/bin/activate
```

## Paso 2: Instalación de Dependencias
Con el entorno virtual activado, instala todas las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

## Paso 3: Configurar las Credenciales

1. Crea un archivo llamado `.env` en la raíz del proyecto.
2. Edita este archivo de texto colocando tu clave de la siguiente forma, sin comillas:

```env
GOOGLE_API_KEY=tu_clave_secreta_aqui
```

## Paso 4: Ejecutar el Proyecto
Para iniciar el asistente interactivo, simplemente ejecuta el archivo principal:

```bash
python main.py
```

### ACLARACIONES

### ¿Qué ocurre al ejecutarlo?
1. **Primera vez:** El sistema detectará que no existe la base de datos local y comenzará la fase algorítmica de indexación (*Chunking* y *Embeddings*). Notarás que el proceso es escalonado y tarda un poco para respetar los límites gratuitos de la API de Google. Se creará automáticamente una carpeta llamada `data_db/`.
2. **Ejecuciones posteriores:** El arranque será inmediato leyendo el almacenamiento estático existente. 
3. Se abrirá la **Consola Interactiva** donde puedes escribir tus consultas y presionar `ENTER`. (Escribe `salir`, `exit`, `quit`, `q` para terminar).

### Mantenimiento de la Base Vectorial
Si agregas nuevos documentos PDF o TXT a la carpeta `/data`, el sistema debe ser reindexado para reconocerlos de la siguiente manera:
1. Elimina por completo la carpeta autogenerada `data_db/`.
2. Vuelve a ejecutar `python main.py` dentro del entorno virtual. El sistema volverá a leer toda la corporación actualizada y fabricará el motor de nuevo.