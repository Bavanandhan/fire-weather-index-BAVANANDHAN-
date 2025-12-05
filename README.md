# Fire Weather Index (FWI) Prediction Model
## Infosys Springboard Machine Learning Project

---

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict the **Fire Weather Index (FWI)** based on environmental data using **Ridge Regression**. The model achieves **98.29% accuracy** on test data with minimal prediction error (MAE: ±1.62 FWI units).

### Key Features:
- ✅ **98.29% R² Score** - Explains 98.29% variance in FWI
- ✅ **Ridge Regression** - Handles multicollinearity effectively
- ✅ **Production-Ready** - Deployed via Flask web application
- ✅ **Real-Time Predictions** - Sub-100ms prediction time
- ✅ **Interactive UI** - Beautiful, responsive web interface
- ✅ **Complete EDA** - Comprehensive data exploration and visualization

---

## 🏗️ Project Structure

```
fwi-prediction-project/
├── app.py                      # Flask application
├── templates/
│   └── index.html             # Web interface
├── ridge.pkl                  # Trained model
├── scaler.pkl                 # Feature scaler
├── feature_cols.pkl           # Feature column names
├── region_mapping.pkl         # Region encoding
├── cleaned_data.csv           # Preprocessed dataset
├── test_predictions.csv       # Model predictions on test set
├── model_performance.csv      # Performance metrics
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── train_model.py            # Training script (reference)
```

---

## 📊 Dataset Information

### Features Used (8 total):
1. **Temperature (°C)** - Range: 0-50°C
2. **Relative Humidity (%)** - Range: 0-100%
3. **Wind Speed (km/h)** - Range: 0-50 km/h
4. **Rain (mm)** - Range: 0-20 mm
5. **FFMC** - Fine Fuel Moisture Code: 0-100
6. **DMC** - Duff Moisture Code: 0-300
7. **ISI** - Initial Spread Index: 0-50
8. **Region** - Bejaia (0) or Sidi-Bel-Abbes (1)

### Target Variable:
- **FWI** - Fire Weather Index: 0-100

### Dataset Statistics:
- **Total Samples**: 500
- **Training Set**: 400 (80%)
- **Test Set**: 100 (20%)
- **Missing Values**: 0
- **Outliers Handled**: Yes (IQR method with 2×IQR)

---

## 🔍 Exploratory Data Analysis (EDA) Results

### Feature Correlations with FWI:
| Feature | Correlation | Relationship |
|---------|-------------|--------------|
| DMC | +0.806 | **Strong Positive** |
| FFMC | +0.415 | Moderate Positive |
| ISI | +0.172 | Weak Positive |
| Wind Speed | +0.151 | Weak Positive |
| Temperature | +0.061 | Very Weak Positive |
| Rain | -0.008 | Negligible |
| Region | -0.045 | Negligible Negative |
| RH (Humidity) | -0.283 | **Moderate Negative** |

### Key Insights:
- **DMC** is the strongest predictor of FWI (explains 64.9% of variance alone)
- **Relative Humidity** has inverse relationship (higher humidity = lower fire risk)
- **FFMC** is the second most important predictor
- Low multicollinearity (only 1 pair with |r| > 0.7)
- Ridge Regression ideal for this dataset

---

## 🤖 Model Architecture

### Algorithm: Ridge Regression

**Why Ridge Regression?**
- Handles multicollinearity through L2 regularization
- Prevents overfitting by penalizing large coefficients
- Balances bias-variance tradeoff

### Model Parameters:
```
Algorithm: Ridge Regression
Optimal Alpha: 0.8302 (selected via 5-fold Cross-Validation)
Alpha Range Tested: 0.01 to 1000 (100 values)
Regularization: L2 penalty
Intercept: 38.2204
```

### Feature Coefficients (Importance):
```
DMC:           +13.26  ⭐⭐⭐ (Most Important)
FFMC:          +7.05   ⭐⭐⭐
RH:            -4.60   ⭐⭐
ISI:           +2.69   ⭐⭐
Temperature:   +1.13   ⭐
Wind Speed:    +0.88   ⭐
Rain:          -0.07   ⭐
Region:        +0.003  ⭐ (Minimal)
```

---

## 📈 Model Performance

### Test Set Performance:
```
Mean Absolute Error (MAE):        1.6154 FWI units
Root Mean Squared Error (RMSE):   2.0807
R² Score:                         0.9829 (98.29%)
```

### Training Set Performance:
```
Mean Absolute Error (MAE):        1.6702 FWI units
Root Mean Squared Error (RMSE):   2.0470
R² Score:                         0.9836 (98.36%)
```

### Generalization Analysis:
- **R² Difference (Train - Test)**: 0.0007 ✅ EXCELLENT
- **Overfitting Risk**: LOW
- **Model Status**: PRODUCTION READY


---

## 📱 Using the Web Application

### User Interface:

**Left Panel - Input Form:**
- Enter environmental data (Temperature, Humidity, etc.)
- Select region from dropdown
- Click "Predict FWI" button

**Right Panel - Results:**
- FWI Score (0-100)
- Risk Level (LOW / MODERATE / HIGH / EXTREME)
- Confidence Level
- Model metrics and accuracy info
- Feature importance visualization

### Risk Levels:
- **LOW (0-10)**: Safe - minimal fire risk
- **MODERATE (10-25)**: Caution - monitor conditions
- **HIGH (25-50)**: Danger - fire risk elevated
- **EXTREME (50+)**: Emergency - high fire risk

### Example Scenarios:

**Scenario 1: Safe Conditions**
```
Temperature: 20°C, Humidity: 80%, Wind: 5 km/h
Rain: 2 mm, FFMC: 30, DMC: 100, ISI: 10
Region: Bejaia
→ Predicted FWI: ~5-10 (LOW RISK)
```

**Scenario 2: Dangerous Conditions**
```
Temperature: 35°C, Humidity: 30%, Wind: 15 km/h
Rain: 0 mm, FFMC: 90, DMC: 200, ISI: 40
Region: Sidi-Bel-Abbes
→ Predicted FWI: ~60-70 (EXTREME RISK)
```

## 📊 Performance Optimization

### Model Tuning:
- Alpha optimized via 5-fold cross-validation
- Tested 100 alpha values from 0.01 to 1000
- Selected alpha=0.8302 for optimal performance

### Prediction Speed:
- Average prediction time: <100ms
- Optimized for real-time inference
- Serialized model (pickle) for fast loading

### Scalability:
- Model can handle 1000+ predictions per second
- Feature scaling ensures numerical stability
- L2 regularization prevents overflow

---

## 📝 Training & Retraining

### Original Training Code:
```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal alpha
ridgecv = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5)
ridgecv.fit(X_train_scaled, y_train)

# Train final model
model = Ridge(alpha=ridgecv.alpha_)
model.fit(X_train_scaled, y_train)

# Save
pickle.dump(model, open('ridge.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

### Retraining Steps:
1. Collect new data with same features
2. Run data cleaning pipeline
3. Apply same preprocessing steps
4. Retrain model with RidgeCV
5. Compare metrics with baseline
6. If improved, update pickle files
7. Test in staging environment
8. Deploy to production


---

## 📚 Educational Resources

### Understanding Ridge Regression:
- Handles multicollinearity through L2 penalty
- Alpha controls regularization strength
- Higher alpha = more regularization (simpler model)
- Lower alpha = less regularization (more complex)

### Feature Engineering:
- Categorical encoding: Label Encoding (Region → 0/1)
- Feature scaling: StandardScaler (mean=0, std=1)
- Outlier handling: IQR method
- No polynomial features needed

### Model Evaluation:
- **R² Score**: 0 to 1 (higher is better)
- **MAE**: Average absolute error (same units as target)
- **RMSE**: Penalizes larger errors more (squared)
- **Train-Test Gap**: Indicates overfitting

---

## 🎯 Next Steps & Enhancements

### Current Implementation:
✅ Ridge Regression model (98.29% accuracy)
✅ Interactive web UI
✅ Real-time predictions
✅ Comprehensive EDA
✅ Production-ready deployment

### Potential Enhancements:
1. **Ensemble Methods**
   - Combine Ridge with other regressors
   - Use Gradient Boosting (XGBoost, LightGBM)
   - Implement stacking or voting

2. **Advanced Features**
   - Time-series forecasting (LSTM/Transformer)
   - Uncertainty quantification
   - Batch predictions API
   - Data versioning with DVC

3. **Monitoring & Logging**
   - Prediction logging
   - Performance drift detection
   - Automatic retraining triggers
   - A/B testing framework

4. **Integration**
   - Real-time weather data feeds
   - Database integration (PostgreSQL)
   - Message queue (Celery)
   - Caching layer (Redis)

5. **Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline (GitHub Actions)
   - Model versioning (MLflow)

---

## 🙏 Acknowledgments

- Dataset: Algerian Forest Fires
- Framework: scikit-learn, Flask
- Deployment: Render, Railway
- Built with: Python, HTML/CSS, JavaScript

---

**Last Updated:** December 5, 2025
**Model Version:** 1.0
**Status:** Production Ready ✅
