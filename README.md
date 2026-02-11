# 🧠 Fuzzy Records Resolution API

## 1. Project Overview

This project provides a **Fuzzy Record Resolution API** designed to detect, cluster, and link duplicate records across large datasets.

The system combines:

- Heuristic blocking strategies
- Distance-based similarity metrics
- Supervised Machine Learning (regularized logistic regression)
- Active learning for model improvement
- Agglomerative clustering for duplicate grouping

It supports model training, dataset deduplication, record linkage, and metadata retrieval via REST API.

---

## 2. Problem Statement

Large datasets (customer records, patient records, CRM exports, etc.) often contain:

- Duplicate entries
- Slight spelling variations
- Inconsistent formatting
- Missing or noisy data

A naive pairwise comparison scales quadratically:

- 42k records → ~880M comparisons
- 1M records → ~500B comparisons

This project addresses scalability and accuracy using blocking, similarity learning, and clustering.

---

## 3. Extraction & Resolution Strategy


The complete fuzzy resolution workflow is summarized below:

| Phase | Component | Description | Output |
|-------|------------|------------|--------|
| 1 | Column Configuration | Define optional, intersected, or customized fields for training | Optimized feature set |
| 2 | Model Training | Train or update model using labeled examples | Trained ML model |
| 3 | Blocking | Apply heuristic predicates (tokens, n-grams, metaphone, numeric, geo rules) | Reduced candidate pairs |
| 4 | Similarity Scoring | Compute Affine Gap and other lexicographic distances | Similarity scores |
| 5 | Supervised Learning | Logistic regression classifies duplicate vs non-duplicate | Labeled record pairs |
| 6 | Active Learning | Low-confidence predictions manually validated and reinjected | Improved model |
| 7 | Clustering | Agglomerative clustering with centroid linkage | Duplicate clusters |
| 8 | API Exposure | REST endpoints for training, deduplication, linkage, metadata | JSON responses |

### Methodological Details

![Combinatorial Explosion Illustration](https://i.ibb.co/Y3t9Fg0/combinations.png)

**Blocking Heuristics**

Next table shows the most performant heuristics used, in order to limit the potential pairs of records to be compared.

| Predicate | Description |
| :---:  | :---: |
| whole Field | common whole string |
| **token** Field | common **entities** of string (i.e. **substrings/words**) |
| first Token | common first _entity_ of string |
| first Two Tokens | common first two _entities_ of string |
| common Integer | common integers |
| alpha Numeric | common numbers in any position |
| near Integers | for any integer N in string, common integers in N-1, N and N+1 positions |
| hundred Integer | integers rounded to nearest hundred |
| hundred Integers Odd | integers rounded to nearest hundred (variant) |
| first Integer | common first integer |
| **ngrams** Tokens | common _entities_ as **n-length sequences of words** |
| common Two Tokens | common two-length _entities_ |
| common Three Tokens | common three-length _entities_ |
| fingerprint | list of sorted words in string |
| one-gram fingerprint | sorted common characters without whitespaces |
| two-gram fingerprint | sorted common two-length _entities_ |
| common four-gram | common consecutive four-length _entities_ |
| common six-gram | common consecutive six-length _entities_ |
| same Three Char Start | common three consecutive characters |
| same Five Char Start | common five consecutive characters |
| same Seven Char Start | common seven consecutive characters |
| suffix Array | list of common suffixes of various lengths |
| sorted Acronym | maximum-length common characters |
| metaphone Token | common phonetic representation (English metaphone) |
| double Metaphone | common double metaphone entities |
| whole Set | whole common set of words |
| common Set Element | common set element |
| common Two Elements | common set of two substrings |
| common Three Elements | common set of three substrings |
| last Set Element | common last substring of set |
| first Set Element | common first substring of set |
| magnitude Of Cardinality | magnitude of cardinality of common entities |
| lat Long Grid | close geographic locations |
| order Of Magnitude | common order of magnitude (log10 geo position) |
| round To l | geographic rounding to precision level |

![Affine Gap Distance Illustration](https://i.ibb.co/VwWH0B2/afine-gap-distance.png)

**Similarity Metric**

Similarity is computed using **Affine Gap distance** (edit-distance variant):

- Counts insertions, deletions and substitutions
- Produces a normalized similarity score
- Used as feature input for the logistic regression classifier

![Active Learning Process](https://i.ibb.co/XkZn64H/active-learning.png)

**Clustering Strategy**

Duplicates are grouped via:
1. Agglomerative clustering
2. Centroid linkage similarity


![Clustering Process](https://i.ibb.co/D1KnDRG/clustering.png)

---

![Complete Methodology Overview](https://i.ibb.co/2qGZ2TZ/complete-approach-methodology.png)

## 4. API Architecture

```text
CSV Dataset (GCS)
      ↓
Data Cleaning (optional)
      ↓
Blocking Predicates
      ↓
Distance-based Similarity
      ↓
Supervised ML (Logistic Regression)
      ↓
Duplicate Labeling
      ↓
Agglomerative Clustering
      ↓
API JSON Response
```

---

## 5. Prerequisites

### General

- Python **3.8+**
- pip
- Access to MongoDB (for model metadata)

### Ubuntu Example

```bash
sudo apt update
sudo apt install python python3-pip
python --version
```

---

## 6. Installation

### Create Virtual Environment

```bash
python -m venv .dedu_env
source .dedu_env/bin/activate   ## for windows : .dedu_env\Scripts\activate.bat
pip install --upgrade pip 		## for windows : .dedu_env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. API Usage

### API Port

```
8080
```

### Train Model

```bash
curl -X POST http://localhost:8080/train-model \
  -H 'Content-Type: application/json' \
  -d '{
    "gcs_path_file": "SELECT_1,9M.csv",
    "trained_model_id": "best",
    "confidence_threshold": 0.85,
    "clean_data": true
  }'
```

### Dataset Deduplication

```bash
curl -X POST http://localhost:8080/dataset-deduplication \
  -H 'Content-Type: application/json' \
  -d '{
    "gcs_path_file": "SELECT_1,9M.csv",
    "trained_model_id": "best",
    "confidence_filter": 0.85,
    "show_statistics": false
  }'
```

### Record to Dataset Linkage

```bash
curl -X POST http://localhost:8080/record-to-dataset-linkage \
  -H 'Content-Type: application/json' \
  -d '{
    "gcs_path_file": "SELECT_1,9M.csv",
    "trained_model_id": "best",
    "record_info": {
      "family_name_list": ["NASSOPOULOS"],
      "first_name_list": ["Georges"]
    }
  }'
```

### Get Models Info

```bash
curl -X GET http://localhost:8080/get-models-info
```

---

## 8. Errors and Exceptions

Possible HTTP errors:

- 400 Bad Request
- 401 Unauthorized
- 404 Not Found
- 408 Timeout
- 429 Too Many Requests
- 503 Service Unavailable
- 520 Unknown Error

Example error format:

```json
{
  "code": "400",
  "details": "JSON parse error",
  "message": "Http Message Not Readable error",
  "timestamp": "string",
  "type": "BAD_REQUEST"
}
```

---

## 9. Tests

Run unit and integration tests:

```bash
pytest -q
```

---

## ✅ Full System Verification (End-to-End)

```bash
# Activate environment
source .dedu_env/bin/activate   ## for windows : .dedu_env\Scripts\activate.bat

# Start API locally
uvicorn src.core.service:app --reload

# In another terminal, call endpoint
curl http://localhost:8080/get-models-info

# Run tests
pytest -q
```

---

## 10. Project Organization

```text
.
├── README.md                             ## Project overview, usage, API description
├── requirements.txt                     ## Python dependencies
├── .env                                 ## Environment variables (not committed)
├── docker/                              ## Docker setup (image + compose)
│   ├── Dockerfile
│   └── docker-compose.yml
├── menu_pipeline.sh                     ## CLI menu to trigger pipeline actions
├── main.py                              ## Application entry point (API launcher)
├── artifacts/                           ## Non-code assets (configs, examples)
│   ├── config/                          ## API & data validation configuration
│   │   ├── swagger.yaml
│   │   └── data_control.json
│   └── examples/                        ## Example API payloads
│       ├── train_model_example_1.json
│       ├── dataset_deduplication_example_2.json
│       └── record_to_dataset_linkage_example_3.json
├── data/                                ## Runtime and training data
│   ├── raw/                             ## Raw input datasets
│   │   ├── faker.csv / faker.json
│   └── active_learning/                 ## Active learning persisted states
│       ├── trained_model_config_1
│       └── variables_predicates_weights_1
├── logs/                                ## Runtime logs
├── tests/                               ## Unit tests
│   └── test_unit.py
└── src/                                 ## Application source code
    ├── core/                            ## Configuration, errors, API service
    │   ├── __init__.py	
    │   ├── config.py
    │   ├── errors.py
    │   ├── service.py
    │   └── controls.py
    ├── model/                           ## Deduplication, fuzzy logic, active learning
    │   ├── __init__.py	
    │   ├── cleaning.py
    │   ├── fuzzy_analysis.py
    │   ├── deduplication.py
    │   └── active_learning.py
    ├── utils/                           ## Shared helpers and logging
    │   ├── __init__.py	
    │   ├── utils.py
    │   └── logging_utils.py
    ├── __init__.py	
    └── pipeline.py

```

---

## 11. Author

Georges Nassopoulos

Professional / Research Project
