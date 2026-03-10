# 🔗 Fuzzy Data Deduplication API

The platform detects, clusters, and links **duplicate records across large datasets** using blocking strategies, similarity learning, and clustering techniques.

---

## 🎯 Project Overview

Main capabilities:

* Detect duplicate records in large datasets
* Perform dataset deduplication
* Link external records to existing datasets
* Train supervised similarity models
* Support active learning for model improvement
* Expose functionality via **REST API**

---

## ⚙️ Tech Stack

* Python
* FastAPI
* Docker & Docker Compose
* MongoDB
* Logistic Regression
* Affine Gap similarity
* Agglomerative clustering
* Active learning

---

## 📂 Project Structure

```text
.
├── README.md                              ## Project overview, usage, API description
├── requirements.txt                       ## Python dependencies
├── .env                                   ## Environment variables
├── menu_pipeline.sh                       ## CLI menu to trigger pipeline actions
├── main.py                                ## Application entry point (API launcher)
│
├── docker/
│   ├── Dockerfile                         ## Application container definition
│   └── docker-compose.yml                 ## Local orchestration
│
├── artifacts/
│   ├── config/
│   │   ├── swagger.yaml                   ## API specification
│   │   └── data_control.json              ## Validation configuration
│   │
│   └── examples/
│       ├── train_model_example_1.json     ## Example payload for training
│       ├── dataset_deduplication_example_2.json
│       └── record_to_dataset_linkage_example_3.json
│
├── data/
│   ├── raw/
│   │   └── faker.csv                      ## Example dataset
│   │
│   └── active_learning/
│       ├── trained_model_config_1         ## Persisted model configuration
│       └── variables_predicates_weights_1 ## Learned predicate weights
│
├── tests/
│   └── test_unit.py                       ## Unit tests
│
└── src/
    ├── pipeline.py                        ## Pipeline orchestration
    │
    ├── core/
    │   ├── config.py                      ## Configuration management
    │   ├── errors.py                      ## Custom exceptions
    │   ├── service.py                     ## FastAPI routes
    │   └── controls.py                    ## Request validation
    │
    ├── model/
    │   ├── cleaning.py                    ## Data cleaning utilities
    │   ├── fuzzy_analysis.py              ## Similarity computation
    │   ├── deduplication.py               ## Deduplication pipeline
    │   └── active_learning.py             ## Active learning workflow
    │
    └── utils/
        ├── utils.py                       ## Shared helpers
        └── logging_utils.py               ## Logging utilities
```

---

## ❓ Problem Statement

Large datasets frequently contain duplicate records due to:

* spelling variations
* inconsistent formatting
* missing fields
* multiple data sources

![Combinatorial Explosion Illustration](https://i.ibb.co/Y3t9Fg0/combinations.png)

Naive pairwise comparison scales quadratically:

* **42k records → ~880M comparisons**
* **1M records → ~500B comparisons**

This project solves scalability and accuracy using **blocking, similarity learning, clustering and active learning **.

---

## 🧠 Approach / Methodology / Strategy

The complete fuzzy resolution workflow is summarized below:

### Extraction & Resolution Strategy

| Phase | Component            | Description                         | Output             |
| ----- | -------------------- | ----------------------------------- | ------------------ |
| 1     | Column Configuration | Define fields used for matching     | Optimized features |
| 2     | Model Training       | Train classifier with labeled pairs | Trained model      |
| 3     | Blocking             | Reduce candidate comparisons        | Candidate pairs    |
| 4     | Similarity Scoring   | Compute distance metrics            | Similarity scores  |
| 5     | Supervised Learning  | Classify duplicates                 | Labeled pairs      |
| 6     | Active Learning      | Improve model with human feedback   | Updated model      |
| 7     | Clustering           | Group duplicates                    | Duplicate clusters |
| 8     | API Exposure         | REST interface                      | JSON responses     |

![Complete Methodology Overview](https://i.ibb.co/2qGZ2TZ/complete-approach-methodology.png)

The project implements the following steps.

#### a) Blocking Heuristics

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

#### b) Similarity Metric

Similarity is computed using **Affine Gap distance** (edit-distance variant):

![Affine Gap Distance Illustration](https://i.ibb.co/VwWH0B2/afine-gap-distance.png)

- Counts insertions, deletions and substitutions
- Produces a normalized similarity score
- Used as feature input for the logistic regression classifier

#### c) Clustering Strategy

Duplicates are grouped via:
1. Agglomerative clustering
2. Centroid linkage similarity

![Clustering Process](https://i.ibb.co/D1KnDRG/clustering.png)

#### d) Active learning

A subset of candidate pairs is presented to the user for validation (accept/reject) to improve the model, **typically before clustering and optionally after if confirmed by the user**.

![Active Learning Process](https://i.ibb.co/XkZn64H/active-learning.png)

---

## 🏗 Pipeline Architecture

```text
CSV Dataset
      ↓
Data Cleaning (optional)
      ↓
Blocking Predicates
      ↓
Similarity Computation
      ↓
Supervised Learning (Logistic Regression)
      ↓
Duplicate Classification
      ↓
Agglomerative Clustering
      ↓
API JSON Response
```

---

## 📊 Exploratory Data Analysis

The platform provides diagnostics such as:

* duplicate cluster distributions
* blocking efficiency metrics
* similarity score distributions

---


## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.8+
* Docker & Docker Compose
* No GPU needed

### 2. OS prerequists

Verify that you have the necessairy packages installed.

#### Windows / WSL2 (recommended)

```bash
# PowerShell
wsl --status
wsl --install
wsl --list --online
wsl --install -d Ubuntu
wsl -d Ubuntu

docker --version
docker compose version
```

#### Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential curl git
python --version
```

### 3. Python environment

```bash
python -m venv .icd10_env
source .dedu_env/bin/activate    	## for windows : .dedu_env\Scripts\activate.bat
pip install --upgrade pip        	## for windows : .dedu_env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

```bash

## Start API server
uvicorn src.core.service:app --reload

## Train model
curl -X POST http://localhost:8080/train-model -H "Content-Type: application/json" -d '{"gcs_path_file":"SELECT_1,9M.csv","trained_model_id":"best","confidence_threshold":0.85,"clean_data":true}'

## Dataset deduplication
curl -X POST http://localhost:8080/dataset-deduplication -H "Content-Type: application/json" -d '{"gcs_path_file":"SELECT_1,9M.csv","trained_model_id":"best","confidence_filter":0.85,"show_statistics":false}'

## Record to dataset linkage
curl -X POST http://localhost:8080/record-to-dataset-linkage -H "Content-Type: application/json" -d '{"gcs_path_file":"SELECT_1,9M.csv","trained_model_id":"best","record_info":{"family_name_list":["NASSOPOULOS"],"first_name_list":["Georges"]}}'

## Get models info
curl -X GET http://localhost:8080/get-models-info

## Run tests
pytest -q
```

---

## 📛 Errors and Exceptions

| Error | Cause | Solution |
|------|------|------|
| 400 Bad Request | Invalid or malformed JSON payload | Verify request body format and required fields |
| 401 Unauthorized | Missing or invalid authentication | Provide valid authentication credentials |
| 404 Not Found | Requested resource or endpoint does not exist | Check API endpoint and request path |
| 408 Timeout | Request processing exceeded time limit | Retry request or reduce dataset size |
| 429 Too Many Requests | API rate limit exceeded | Wait before sending additional requests |
| 503 Service Unavailable | API service temporarily unavailable | Verify server status and retry later |
| 520 Unknown Error | Unexpected server-side error | Check server logs for more details |

Example response:

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

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Fuzzy Analysis / Data Quality AI Project
