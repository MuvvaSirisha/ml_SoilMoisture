# 🌍 A.V.A.N.I. — Soil Moisture Intelligence Engine

**A.V.A.N.I.** is an AI-powered soil moisture analysis platform built for India. It combines satellite data, local LLMs, Google Earth Engine, and scientific literature into a conversational Streamlit interface.

---

## ✨ Features

- 🗣️ **Natural language queries** — Ask in plain English (e.g. *"Show soil moisture trend in Punjab from 2015 to 2020"*)
- 📊 **Analysis operations** — Mean, trend, comparison, seasonal patterns
- 🗺️ **Spatial maps** — State-level and India-wide choropleth maps with real shapefile borders
- ☁️ **GEE SMAP tab** — Live NASA SMAP 9km data via Google Earth Engine
- 📚 **Literature Q&A** — Upload scientific PDFs and query them with vision AI
- 🛡️ **Safety guardrails** — On-topic query filtering

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Guardrails → Intent Classifier → Query Classifier
                                        │
                              ┌─────────▼──────────┐
                              │     SM Engine       │
                              │  (engine.py)        │
                              │                     │
                              │  ┌───────────────┐  │
                              │  │ Zarr Dataset  │  │
                              │  │ (Google Drive)│  │
                              │  └───────────────┘  │
                              │  ┌───────────────┐  │
                              │  │  Shapefile    │  │
                              │  │ (Google Drive)│  │
                              │  └───────────────┘  │
                              └─────────────────────┘
                                        │
                                   Streamlit UI
```

---

## ⚙️ Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| [Ollama](https://ollama.com) | Local LLM for NLP | `winget install Ollama.Ollama` |
| Git | Version control | [git-scm.com](https://git-scm.com) |

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml_SoilMoisture.git
cd ml_SoilMoisture
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
# Text model (for query understanding)
ollama pull qwen2.5:3b

# Vision model (for literature Q&A with diagrams)
ollama pull minicpm-v:latest
```

Make sure Ollama is **running in the background** before starting the app.

### 5. Set up credentials

#### Google Drive (Service Account)
The app uses a Google Service Account to download datasets from Google Drive.

1. Create a project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Google Drive API**
3. Create a **Service Account** and download its JSON key
4. Place the key file at `cloud/service_account.json`
5. Share your Google Drive dataset folders with the service account email

#### Environment variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your values
GOOGLE_SERVICE_ACCOUNT_KEY=cloud/service_account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

#### Google Earth Engine (for GEE SMAP tab)

```bash
# One-time authentication
earthengine authenticate
```

Then enable the Earth Engine API for your GCP project at:
https://console.cloud.google.com/apis/library/earthengine.googleapis.com

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **First run note:** The app will automatically download Zarr datasets and shapefiles from Google Drive into `cache/`. This may take a few minutes depending on your internet speed. Subsequent runs use the local cache.

---

## 📁 Project Structure

```
ml_SoilMoisture/
│
├── app.py                  # Main Streamlit UI
├── main.py                 # CLI interface & shared helpers
├── engine.py               # Soil moisture analysis engine
├── Config.py               # Configuration (Ollama URLs, settings)
│
├── Query_classifier.py     # Classifies user queries into operations
├── intent_classifier.py    # Determines query intent (analysis/literature/chat)
├── agent.py                # Ollama LLM agent for query clarification
├── guardrails.py           # Safety filter for off-topic queries
│
├── gee_smap.py             # Google Earth Engine SMAP integration
├── literature_manager.py   # PDF/DOCX literature management
├── literature_qa.py        # Literature Q&A with vision model
├── vision_q.py             # Vision model interface (minicpm-v)
├── utils.py                # Shared utilities
│
├── cloud/                  # Google Drive integration scripts
│   ├── dataset_manager.py  # Download/cache Zarr datasets
│   ├── shapefile_manager.py# Download/cache shapefiles
│   ├── download_zarr.py    # Bulk download utility (run once)
│   ├── list_drive_files.py # List Drive contents (utility)
│   └── literature_manager.py # Sync literature PDFs from Drive
│
├── .env.example            # Template for environment variables
├── .gitignore              # Protects secrets and cache from Git
└── requirements.txt        # Python dependencies
```

---

## 🔐 Security

The following files are **gitignored** and must **never be committed**:

| File | Why |
|---|---|
| `.env` | Contains your GCP project ID |
| `cloud/service_account.json` | Google Service Account private key |
| `cache/` | Large data files (downloaded from Drive) |

Always use `.env` for secrets. Never hardcode credentials.

---

## 🧰 Data Pipeline

```
Google Drive (your datasets)
       │
       ▼ cloud/dataset_manager.py
       │  → Downloads YYYY_v2.zarr.zip for each year
       │  → Extracts to cache/zarr/
       │
       ▼ cloud/shapefile_manager.py
       │  → Downloads India state shapefile components
       │  → Saves to cache/shapefiles/
       │
       ▼ engine.py (SM_Engine)
          → Loads xarray dataset from cache
          → Loads GeoDataFrame from shapefile
          → Runs requested analysis
          → Returns charts and statistics
```

---

## 📊 Supported Operations

| Operation | Description |
|---|---|
| `mean` | Average soil moisture for a period |
| `anomaly` | Deviation from long-term average |
| `trend` | Linear trend over time |
| `comparison` | Compare two regions or time periods |
| `seasonal` | Monthly/seasonal patterns |

---

## 💬 Example Queries

```
"Show soil moisture trend in Maharashtra from 2015 to 2022"
"Compare Punjab and Haryana soil moisture in 2019"
"What is the seasonal pattern in Rajasthan?"
"Summarise findings from the uploaded paper"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is for academic and research purposes.

---

## 👩‍💻 Author

Built as part of soil moisture ML research using NASA SMAP satellite data for India.
