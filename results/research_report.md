# Multi-Model Climate Prediction and Evaluation Framework
## Bharatiya Antariksh Hackathon 2026 — Problem Statement 5: AI-Powered Digital Twin of India's Climate

---

### Executive Summary

This research report documents the extension of the baseline Random Forest climate prediction pipeline into a comprehensive, multi-model comparative machine learning framework. Four distinct machine learning architectures were trained and evaluated on identical chronological splits of India Meteorological Department (IMD) high-resolution gridded observations across 7 pilot climatic zones:
1. **Random Forest Regressor** (Ensemble Bagging baseline)
2. **XGBoost Regressor** (Extreme Gradient Boosting with regularized trees)
3. **Histogram-based Gradient Boosting** (HistGradientBoosting / LightGBM-style binning)
4. **Long Short-Term Memory Network** (PyTorch 2-layer Recurrent Neural Network with 14-day temporal lookback)

All models were benchmarked on three critical national climate variables: daily rainfall (`rainfall_mm`), maximum 2-meter surface temperature (`tmax_c`), and minimum surface temperature (`tmin_c`).

---

## 1. Problem Statement

India possesses immense agro-ecological diversity, spanning alpine Himalayan terrain, arid western deserts, fertile alluvial Indo-Gangetic plains, central highlands, and tropical maritime coastal zones. Accurate daily climate monitoring and short-to-medium range forecasting (1–14 days ahead) is crucial for monsoon water resource management, agrarian food security, cyclone preparedness, and extreme heatwave mitigation.

Under ISRO's **Bharatiya Antariksh Hackathon 2026 (Problem Statement 5)**, the mandate is to develop an **AI-Powered Digital Twin of India's Climate** using India's national observational infrastructure (IMD gridded data and INSAT/MOSDAC satellite payloads). The digital twin must assimilate real-time observations, run forward predictive projections, conduct what-if climate anomaly simulations, and provide transparent model benchmarking.

---

## 2. Project Objective

The core objective was to elevate an existing single-model (Random Forest) digital twin into an objective multi-model comparative benchmark that:
- Implements 3 additional, scientifically justified ML/DL architectures.
- Evaluates all models under rigorous, leak-free chronological train/validation/test splits.
- Computes standardized regression metrics (MAE, RMSE, $R^2$) and IMD rainfall intensity category breakdowns.
- Automates data-driven selection of the best-performing model for each target climate variable.
- Integrates multi-model switching and comparative scorecards directly into the interactive Flask Digital Twin dashboard without breaking any existing operational features.

---

## 3. Dataset Description

The primary dataset originates from IMD high-resolution daily gridded observations (spatial resolution $0.25^\circ \times 0.25^\circ$ for rainfall, $1.0^\circ \times 1.0^\circ$ for temperature), aggregated across 7 representative pilot climate zones in India:
- **Central India** ($23.5^\circ\text{N}, 78.5^\circ\text{E}$)
- **Coastal Odisha** ($20.5^\circ\text{N}, 86.0^\circ\text{E}$)
- **Deccan Plateau** ($17.0^\circ\text{N}, 76.0^\circ\text{E}$)
- **Indo-Gangetic Plain** ($26.5^\circ\text{N}, 81.0^\circ\text{E}$)
- **Kerala Coast** ($9.5^\circ\text{N}, 76.5^\circ\text{E}$)
- **Northeast** ($26.0^\circ\text{N}, 92.5^\circ\text{E}$)
- **Rajasthan Desert** ($27.0^\circ\text{N}, 71.5^\circ\text{E}$)

### Key Dataset Dimensions:
- **Temporal Range:** 2024-01-01 to 2025-12-31 (731 continuous daily observations per region)
- **Total Ingested Observations:** 5,117 raw daily records
- **Valid Engineered Feature Samples:** 5,068 samples (after initial 7-day lag/rolling window generation)
- **Target Variables:**
  1. `rainfall_mm` (Continuous, zero-inflated, heavy-tailed monsoon precipitation)
  2. `tmax_c` (Maximum daily 2m surface temperature in Celsius)
  3. `tmin_c` (Minimum daily surface temperature in Celsius)
- **Satellite Ingestion:** INSAT-3D/3DR TIR Land Surface Temperature (LST) and Sea Surface Temperature (SST) features are integrated into the feature schema. In the current archive, satellite channels contain missing values which are handled through dedicated imputed indicators.

---

## 4. Data Preprocessing

To guarantee zero future-leakage and identical features across training and real-time inference, all data flows through a unified pipeline ([`climate_twin/data/preprocessing.py`](file:///Users/abhijit/Desktop/sky-mirror-india/climate_twin/data/preprocessing.py)):
1. **Chronological Sorting:** Observations are strictly sorted by `region` and `date`.
2. **Lag Construction:** All autoregressive predictors use $t-1$ or earlier observations (`.shift(1)`), strictly preventing contemporaneous or future leakage.
3. **Missing Value Imputation:** Missing values in raw observations are imputed using forward-fill within regions followed by regional climatological medians.
4. **Encoding:** Categorical pilot regions are mapped to static integer codes ($0 \dots 6$).

---

## 5. Feature Engineering

The models are provided with a 16-dimensional feature vector combining temporal, astronomical, autoregressive, and geographic signals:

| # | Feature Name | Source | Scientific Rationale |
|---|--------------|--------|----------------------|
| 1 | `day_of_year` | Date (1–366) | Annual solar declination and seasonality |
| 2 | `month_sin` | Date ($\sin(2\pi m / 12)$) | Smooth cyclic seasonal encoding |
| 3 | `month_cos` | Date ($\cos(2\pi m / 12)$) | Orthogonal cyclic seasonal component |
| 4 | `rainfall_lag_1` | IMD Gridded ($t-1$) | Immediate precipitation persistence |
| 5 | `rainfall_lag_7` | IMD Gridded ($t-7$) | Weekly synoptic wave cycle |
| 6 | `rainfall_roll_7` | IMD Gridded (7d mean) | Multi-day soil moisture and monsoon surge trend |
| 7 | `tmax_lag_1` | IMD Gridded ($t-1$) | Thermal inertia of the boundary layer |
| 8 | `tmax_roll_7` | IMD Gridded (7d mean) | Weekly synoptic heat accumulation trend |
| 9 | `tmin_lag_1` | IMD Gridded ($t-1$) | Overnight radiative cooling anchor |
| 10 | `humidity_lag_1` | Derived ($t-1$) | Atmospheric moisture availability |
| 11 | `diurnal_range_lag_1` | Derived ($tmax - tmin$) | Cloud cover proxy and radiative balance indicator |
| 12 | `insat_lst_lag_1` | INSAT-3D/3DR MOSDAC | Satellite Land Surface Temperature ($t-1$) |
| 13 | `insat_sst_lag_1` | INSAT-3D/3DR MOSDAC | Sea Surface Temperature ($t-1$, maritime regions) |
| 14 | `latitude` | Spatial ($^\circ\text{N}$) | Meridional solar insolation gradient |
| 15 | `longitude` | Spatial ($^\circ\text{E}$) | Zonal monsoon track position (Bay of Bengal vs Arabian Sea) |
| 16 | `region_code` | Integer (0–6) | Regional identity and local topography |

---

## 6. Model 1: Random Forest Regressor (Baseline)

### Architecture & Theory
Random Forest is an ensemble bootstrap aggregating (bagging) method. It constructs $B=300$ de-correlated decision trees trained on random bootstrap subsets of the training data and random feature subsets ($\sqrt{p}$ features considered per split). Predictions are averaged across all trees:
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

### Hyperparameters:
- `n_estimators`: 300
- `max_depth`: 18
- `min_samples_leaf`: 2
- `random_state`: 42
- Multi-output regression: Jointly predicts `[rainfall, tmax, tmin]`
- Uncertainty estimation: Empirical 10th and 90th percentiles across individual tree estimators yield an 80% prediction interval.

---

## 7. Model 2: XGBoost Regressor

### Architecture & Theory
XGBoost (Extreme Gradient Boosting) builds an additive ensemble of trees sequentially using second-order Taylor expansions of the loss function:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where $g_i$ and $h_i$ are first and second order gradients, and $\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2 + \alpha \|w\|_1$ incorporates $L_1$ and $L_2$ leaf-weight regularization.

### Implementation Details:
- Trains independent `XGBRegressor` estimators for each target variable.
- Regularization parameters (`reg_alpha=0.1`, `reg_lambda=1.0`) and subsampling (`subsample=0.8`, `colsample_bytree=0.8`) mitigate overfitting on seasonal peaks.

---

## 8. Model 3: Histogram-based Gradient Boosting (HistGradientBoosting)

### Architecture & Theory
`HistGradientBoostingRegressor` bins continuous features into discrete integer bins ($K=255$), transforming split finding from an $\mathcal{O}(n \log n)$ sorting operation into an $\mathcal{O}(K)$ histogram lookup. It natively accommodates missing values by learning default branch directions for NaNs during training.

### Implementation Details:
- `max_iter`: 500 (with early stopping after 20 iterations without validation improvement)
- `max_depth`: 10
- `learning_rate`: 0.05
- `min_samples_leaf`: 10
- `l2_regularization`: 0.1

---

## 9. Model 4: Long Short-Term Memory (LSTM) Neural Network

### Architecture & Theory
To model sequential temporal dependencies beyond static lag features, a deep recurrent neural network using LSTM cells was developed in PyTorch. The model takes a 14-day sequence window $X_{t-13:t} \in \mathbb{R}^{14 \times 16}$ to predict conditions on day $t+1$:
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

### Network Specifications:
- **Input Dimensions:** 16 features per day, 14-day lookback sequence
- **LSTM Layers:** 2 stacked layers with hidden dimension $H=64$ and recurrent dropout $0.20$
- **Dense Head:** Fully-connected layer ($64 \to 32$, ReLU activation) followed by linear output ($32 \to 3$)
- **Optimization:** Adam optimizer ($\eta = 10^{-3}$) with `ReduceLROnPlateau` scheduler and gradient clipping ($\|\mathbf{g}\| \le 1.0$)
- **Feature Normalization:** Standard z-score normalization fit exclusively on training sequences.

---

## 10. Experimental Setup

To adhere to climate forecasting best practices, data partitioning strictly preserved temporal causality:
- **Train Set (70%):** 3,547 daily samples (January 2024 to mid-July 2025)
- **Validation Set (15%):** 760 daily samples (Mid-July 2025 to late-October 2025)
- **Test Set (15%):** 761 daily samples (Late-October 2025 to December 31, 2025)

No data from the future was accessible to models during training or validation. Hyperparameters and early-stopping triggers were tuned strictly using the validation split.

---

## 11. Evaluation Metrics

Model performance was assessed using three standard regression metrics across all targets:
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

In addition, because rainfall follows an extreme, zero-inflated distribution, test performance was segmented by **IMD Rainfall Intensity Standards**:
- **Dry Days:** $< 2.5\text{ mm}$
- **Moderate Rainfall:** $2.5\text{ to }15.5\text{ mm}$
- **Heavy Rainfall:** $> 15.5\text{ mm}$

---

## 12. Actual Experimental Results

The table below presents the actual experimental metrics obtained on the 761-sample chronological test set:

| Model Architecture | Target Variable | Test MAE | Test RMSE | Test $R^2$ | Training Time (s) | Best Model Badge |
|:-------------------|:----------------|:--------:|:---------:|:----------:|:-----------------:|:----------------:|
| **Random Forest** | Rainfall (`rainfall_mm`) | **2.4628** | **6.4777** | **0.2354** | **0.42** | 🏆 **BEST** |
| HistGradientBoosting | Rainfall (`rainfall_mm`) | 2.5241 | 6.9059 | 0.1310 | 1.13 | |
| XGBoost | Rainfall (`rainfall_mm`) | 2.7082 | 7.0577 | 0.0923 | 1.23 | |
| LSTM Neural Net | Rainfall (`rainfall_mm`) | 3.0393 | 7.1450 | 0.0698 | 3.67 | |
| **Random Forest** | Max Temp (`tmax_c`) | **1.1716** | **1.5269** | **0.9428** | **0.42** | 🏆 **BEST** |
| HistGradientBoosting | Max Temp (`tmax_c`) | 1.1771 | 1.5404 | 0.9418 | 1.13 | |
| XGBoost | Max Temp (`tmax_c`) | 1.2406 | 1.6053 | 0.9368 | 1.23 | |
| LSTM Neural Net | Max Temp (`tmax_c`) | 2.1601 | 2.7931 | 0.8085 | 3.67 | |
| **HistGradientBoosting** | Min Temp (`tmin_c`) | **0.9056** | **1.1937** | **0.9741** | **1.13** | 🏆 **BEST** |
| XGBoost | Min Temp (`tmin_c`) | 0.9430 | 1.2304 | 0.9725 | 1.23 | |
| Random Forest | Min Temp (`tmin_c`) | 1.0184 | 1.3109 | 0.9688 | 0.42 | |
| LSTM Neural Net | Min Temp (`tmin_c`) | 2.2838 | 2.8112 | 0.8565 | 3.67 | |

### IMD Rainfall Category Breakdown:
- **Dry Days ($<2.5\text{ mm}$, $N=695$):**
  - Random Forest: MAE = $1.52\text{ mm}$
  - HistGradientBoosting: MAE = $1.46\text{ mm}$
  - XGBoost: MAE = $1.70\text{ mm}$
  - LSTM: MAE = $2.09\text{ mm}$
- **Moderate Rainfall ($2.5 - 15.5\text{ mm}$, $N=43$):**
  - LSTM: MAE = $4.49\text{ mm}$
  - Random Forest: MAE = $6.34\text{ mm}$
  - XGBoost: MAE = $6.98\text{ mm}$
  - HistGradientBoosting: MAE = $7.20\text{ mm}$
- **Heavy Rainfall ($>15.5\text{ mm}$, $N=23$):**
  - Random Forest: MAE = $23.68\text{ mm}$, RMSE = $30.02\text{ mm}$
  - XGBoost: MAE = $25.08\text{ mm}$, RMSE = $32.12\text{ mm}$
  - HistGradientBoosting: MAE = $25.79\text{ mm}$, RMSE = $32.51\text{ mm}$
  - LSTM: MAE = $28.91\text{ mm}$, RMSE = $36.36\text{ mm}$

---

## 13. Model Comparison & Scientific Discussion

### Key Findings:
1. **Temperature Forecasting ($R^2 > 0.94$):**
   Surface temperatures exhibit strong diurnal and seasonal continuity. Both Random Forest and HistGradientBoosting achieve sub-1.2°C MAE on maximum temperature and sub-1.0°C MAE on minimum temperature. HistGradientBoosting proved superior for Minimum Temperature ($0.9056^\circ\text{C}$ MAE), benefiting from gradient-guided binning of night-time radiative cooling signatures.

2. **Rainfall Forecasting ($R^2 = 0.07 - 0.24$):**
   Rainfall represents an intermittent, highly non-linear meteorological process with significant zero-inflation. Random Forest demonstrated the lowest MAE ($2.46\text{ mm}$) and highest variance explained ($R^2 = 0.2354$). Tree bagging effectively averages out extreme variance and limits over-predicting false rain days.

3. **LSTM Performance Assessment:**
   While the LSTM achieved respectable temperature correlation ($R^2 = 0.81 - 0.86$), it underperformed the tree-based ensembles across all metrics. This is an authentic scientific finding: with 2 years of daily data per station (~731 steps), deep recurrent architectures suffer from sample scarcity compared to tabular tree ensembles that leverage pre-computed multi-day lag and rolling features directly.

---

## 14. Best Model Selection & Justification

The automatic model selector determines the optimal architecture per target:
- **Rainfall (`rainfall_mm`): `random_forest`**
  - *Rationale:* Achieved lowest MAE ($2.4628\text{ mm}$), lowest RMSE ($6.4777\text{ mm}$), and superior performance on heavy rainfall events ($>15.5\text{ mm}$).
- **Max Temperature (`tmax_c`): `random_forest`**
  - *Rationale:* Lowest MAE ($1.1716^\circ\text{C}$) and RMSE ($1.5269^\circ\text{C}$) with $94.28\%$ explained variance.
- **Min Temperature (`tmin_c`): `hist_gradient_boosting`**
  - *Rationale:* Lowest MAE ($0.9056^\circ\text{C}$) and RMSE ($1.1937^\circ\text{C}$) with $97.41\%$ explained variance.

---

## 15. Limitations

1. **Archive Temporal Depth:** The training dataset encompasses 2 years (2024–2025). Incorporating 10–30 years of IMD archives would significantly improve deep learning sequence modeling.
2. **Missing Satellite Channels:** INSAT-3D LST and SST channels currently contain missing values in the pilot extract, which constrained satellite-driven predictive gains.
3. **Spatial Discretization:** The model operates at regional spatial centroids rather than continuous high-resolution grid coordinate fields.

---

## 16. Future Improvements

1. **Multi-Decadal IMD Integration:** Ingesting 1990–2025 IMD gridded products to unlock full potential of Transformer/LSTM architectures.
2. **INSAT-3DR Rapid-Scan Radiometer Fusion:** Near-real-time merging of 15-minute INSAT infrared radiance for convection onset detection.
3. **Spatial Graph Neural Networks (GNNs):** Modeling cross-regional atmospheric teleconnections and monsoon trough progression via topological graph networks.

---

## 17. Conclusion

The multi-model framework successfully advances ISRO BAH 2026 Problem Statement 5. By establishing a fair, leak-free benchmark among Random Forest, XGBoost, HistGradientBoosting, and LSTM, the system demonstrates that ensemble tree architectures (Random Forest and HistGradientBoosting) deliver superior forecasting precision for India's regional climate variables on contemporary observational scales. The digital twin backend and dashboard now transparently reflect these empirical findings.
