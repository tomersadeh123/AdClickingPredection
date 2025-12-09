# Ad CTR Prediction - Development Plan

## Project Goal
Build a production-ready ML system that demonstrates full ML lifecycle expertise for the Taboola AI Solution Engineer role.

## Timeline: 10-14 Days

### Phase 1: Data & Exploration (Days 1-2)
- [x] Set up project structure
- [ ] Download Avazu CTR dataset from Kaggle
- [ ] Exploratory data analysis (EDA) notebook
- [ ] Data quality checks
- [ ] Class imbalance analysis

### Phase 2: Feature Engineering (Days 3-4)
- [ ] Feature engineering pipeline
- [ ] Categorical encoding (hash encoding for high cardinality)
- [ ] Time-based features (hour, day of week)
- [ ] Feature selection and importance
- [ ] Handle missing values

### Phase 3: Model Training (Days 5-7)
- [ ] Baseline: Logistic Regression (scikit-learn)
- [ ] Model 2: XGBoost
- [ ] Model 3: Neural Network (PyTorch)
- [ ] MLflow experiment tracking
- [ ] Hyperparameter tuning with Optuna
- [ ] Model comparison and selection

### Phase 4: MLOps & Deployment (Days 8-10)
- [ ] FastAPI model serving
- [ ] Model versioning and registry
- [ ] Performance monitoring
- [ ] Docker containerization
- [ ] Deploy to Railway/Render

### Phase 5: Documentation & Polish (Days 11-14)
- [ ] Comprehensive README
- [ ] Architecture diagrams
- [ ] Performance metrics dashboard
- [ ] Interview preparation doc
- [ ] GitHub polish

## Key Deliverables

### Technical
✅ Data preprocessing pipeline
✅ 3 trained models with comparison
✅ MLflow experiment tracking
✅ REST API for predictions
✅ Dockerized deployment
✅ Performance monitoring

### Documentation
✅ Technical README
✅ API documentation
✅ Model performance report
✅ Architecture diagram
✅ Interview talking points

## Success Metrics

- **Model Performance**: AUC-ROC > 0.75
- **API Latency**: < 100ms per prediction
- **Code Quality**: Clean, documented, tested
- **MLOps**: Full experiment tracking and versioning
- **Interview Ready**: Can explain every design decision

## Dataset: Avazu Click-Through Rate Prediction

**Source**: Kaggle - https://www.kaggle.com/c/avazu-ctr-prediction

**Size**: 40M+ rows, 24 features

**Features**:
- `click` (target): 0 or 1
- `hour`: YYMMDDHH format
- `C1`: Categorical feature
- `banner_pos`: Ad position
- `site_id`, `site_domain`, `site_category`: Site features
- `app_id`, `app_domain`, `app_category`: App features
- `device_id`, `device_type`, `device_model`: Device features
- And more...

**Challenge**: High cardinality categorical features (millions of unique values)

## Tech Stack Justification

**PyTorch** (Required by job):
- Neural network model training
- Custom architectures
- Production deployment

**scikit-learn** (Required by job):
- Baseline models (Logistic Regression)
- Preprocessing pipelines
- Model evaluation metrics

**XGBoost**:
- Industry standard for CTR prediction
- Handles categorical features well
- Fast training and inference

**MLflow**:
- Experiment tracking
- Model versioning
- Parameter logging
- MLOps best practices

**FastAPI**:
- Modern, fast API framework
- Automatic API docs
- Async support
- Production-ready

## Interview Talking Points

### Problem Solving
"I chose CTR prediction because it's directly relevant to Taboola's business. Ad platforms need accurate CTR predictions for bidding, placement, and revenue optimization."

### Technical Decisions
"I used hash encoding for high-cardinality categorical features instead of one-hot encoding because with millions of unique site_ids and device_ids, one-hot encoding would create an unmanageable feature space."

### MLOps
"I tracked all experiments with MLflow, logging parameters, metrics, and artifacts. This enables reproducibility and makes it easy to compare models and roll back if needed."

### Production Thinking
"I built a FastAPI service with Pydantic validation, containerized it with Docker, and deployed to Railway with health checks and monitoring endpoints."

### Business Impact
"Improving CTR prediction accuracy by even 1-2% can significantly increase ad revenue. If Taboola serves billions of impressions, small improvements compound to millions in additional revenue."
