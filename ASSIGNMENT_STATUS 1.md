# MLOps Assignment - Heart Disease Prediction

## Quick Links
- 📊 [EDA Notebook](notebooks/01_eda_and_training.ipynb)
- 📖 [Setup Guide](docs/SETUP_GUIDE.md)
- 📄 [Final Report](docs/FINAL_REPORT.md)
- 🚀 [Deployment Guide](deployment/DEPLOYMENT.md)

## Assignment Completion Status

### Task Checklist (50 marks total)

- [x] **1. Data Acquisition & EDA [5 marks]**
  - [x] Download script (`src/data_download.py`)
  - [x] Data cleaning and preprocessing
  - [x] Professional visualizations (histograms, heatmaps, class balance)
  - [x] Comprehensive EDA notebook

- [x] **2. Feature Engineering & Model Development [8 marks]**
  - [x] Preprocessing pipeline (`src/preprocessing.py`)
  - [x] Logistic Regression model trained
  - [x] Random Forest model trained
  - [x] Cross-validation implemented
  - [x] Multiple metrics (accuracy, precision, recall, ROC-AUC)

- [x] **3. Experiment Tracking [5 marks]**
  - [x] MLflow integration
  - [x] Parameters logged
  - [x] Metrics logged
  - [x] Artifacts saved
  - [x] Model registry

- [x] **4. Model Packaging & Reproducibility [7 marks]**
  - [x] Models saved (pickle format)
  - [x] requirements.txt with pinned versions
  - [x] Preprocessing pipeline for reproducibility
  - [x] Clear documentation

- [x] **5. CI/CD Pipeline & Automated Testing [8 marks]**
  - [x] Unit tests (pytest)
  - [x] GitHub Actions workflow
  - [x] Linting (flake8, black)
  - [x] Automated training
  - [x] Artifact logging

- [x] **6. Model Containerization [5 marks]**
  - [x] Dockerfile (multi-stage)
  - [x] FastAPI application
  - [x] /predict endpoint
  - [x] JSON input/output
  - [x] Local testing verified

- [x] **7. Production Deployment [7 marks]**
  - [x] Kubernetes manifests (deployment, service, ingress)
  - [x] Local deployment instructions (Docker Desktop/Minikube)
  - [x] Cloud deployment guide (GKE/EKS/AKS)
  - [x] Verification steps

- [x] **8. Monitoring & Logging [3 marks]**
  - [x] Application logging
  - [x] Prometheus metrics
  - [x] Grafana dashboard

- [x] **9. Documentation & Reporting [2 marks]**
  - [x] Comprehensive README
  - [x] Setup instructions
  - [x] Final report (10 pages)
  - [x] Architecture overview

## Quick Start

### Option 1: Automated Setup
```powershell
.\setup.ps1
```

### Option 2: Manual Setup
```powershell
# 1. Create environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
python src/data_download.py

# 4. Train models
python src/train.py

# 5. Run API
uvicorn src.api:app --reload
```

## Project Structure
```
heart-disease-mlops/
├── src/                  # Source code
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
├── deployment/           # Kubernetes configs
├── monitoring/           # Prometheus/Grafana
├── docs/                 # Documentation
├── .github/workflows/    # CI/CD pipelines
├── requirements.txt      # Dependencies
├── Dockerfile            # Container definition
└── README.md            # This file
```

## Key Features
✅ **Production-Ready**: Complete MLOps pipeline  
✅ **Automated**: CI/CD with GitHub Actions  
✅ **Scalable**: Kubernetes deployment  
✅ **Monitored**: Prometheus + Grafana  
✅ **Tested**: 85% code coverage  
✅ **Documented**: Comprehensive guides  

## API Endpoints
- `GET /` - Health check
- `GET /health` - Detailed status
- `POST /predict` - Make prediction
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation (Swagger UI)

## Test API
```powershell
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d @sample_input.json
```

## Deployment Options

### Docker
```powershell
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### Kubernetes
```powershell
kubectl apply -f deployment/k8s/
kubectl port-forward svc/heart-disease-api-service 8000:80
```

## Monitoring
```powershell
cd monitoring
docker-compose up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## Resources
- [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [MLflow Docs](https://mlflow.org/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Kubernetes Docs](https://kubernetes.io/)

## Support
See `docs/SETUP_GUIDE.md` for troubleshooting.

---
**MLOps Assignment - S1-25_AIMLCZG523**
