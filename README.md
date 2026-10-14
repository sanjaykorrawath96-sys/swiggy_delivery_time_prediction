# swiggy_delivery_time_prediction
# 🛵 Swiggy Delivery Time Prediction

A machine learning project that predicts **how many minutes a food order will take to be delivered**, using rider, order, weather, traffic, city and distance information. It follows the **CRISP-ML(Q)** process, compares six regression models, validates the best one with **5-fold cross-validation**, tunes it with **Optuna**, and saves the final pipeline for use in a **Streamlit** app.

## Business Problem
Inaccurate delivery ETAs reduce customer trust, increase cancellations and refunds, raise support volume and lead to poor rider allocation. Delivery time is affected by traffic, weather, restaurant preparation, rider behaviour, batched orders and city conditions. This project builds a model that predicts delivery time (`time_taken`) so a reliable ETA can be shown at checkout.

## Final Result

- **Task:** Regression (target: `time_taken` in minutes)
- **Best Model:** XGBoost Regressor (Optuna-tuned)
- **Validation:** 80/20 hold-out split + 5-fold cross-validation
- **Hyperparameter Tuning:** Optuna, 50 trials
- **R² Score (test data):** 0.831
- **MAE:** ≈ 3.08 minutes
- **RMSE:** ≈ 3.88 minutes
- **MSE:** ≈ 15.03
- **In plain words:** the model is off by about 3 minutes on average and explains about 83% of the variation in delivery time on unseen orders.

The model is off by about 3 minutes on average and explains about 83% of the variation in delivery time on unseen orders.

## Tools & Libraries
- **Python:** core programming language
- **Jupyter Notebook:** EDA, experimentation and model building
- **pandas:** loading the CSV, info/describe, duplicate check, dropping columns, missing-value counts
- **NumPy:** numerical operations
- **Matplotlib:** plot layout and figures
- **Seaborn:** correlation heatmap, KDE plots and box plots
- **KNNImputer (scikit-learn):** fills missing numeric values using nearest neighbours
- **SimpleImputer (scikit-learn):** fills missing categorical values with the most frequent category
- **StandardScaler:** standardises numeric features
- **PowerTransformer:** reduces skewness of numeric features
- **OneHotEncoder:** encodes nominal (unordered) categories
- **OrdinalEncoder:** encodes ordered categories (traffic, city type)
- **Pipeline & ColumnTransformer:** chain preprocessing and the model, and apply different steps to different column groups
- **train_test_split:** 80% train / 20% test split
- **KFold & cross_val_score:** 5-fold cross-validation
- **LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, SVR, RandomForestRegressor:** baseline and comparison models
- **XGBoost (XGBRegressor):** final selected model
- **MSE, MAE, RMSE, R²:** model evaluation metrics
- **Optuna:** Bayesian hyperparameter optimisation
- **pickle:** saves the final pipeline as a `.pkl` file
- **Streamlit:** web app that loads the saved model for predictions
- **Git & GitHub:** hosting and sharing the project

## Dataset
- **File:** `swiggy_demographic.csv` with 45,502 rows and 26 columns (19 kept for modelling); no duplicate rows.
- **Numeric features:** age, ratings, vehicle condition, multiple deliveries, order day, order month, is weekend, pickup time, order hour, distance.
- **Nominal features:** weather, vehicle type, festival, day of week, time of day.
- **Ordinal features:** traffic (low < medium < high < jam) and city type (semi-urban < urban < metropolitian).
- **Missing values** were present in 10 columns (largest: distance, ratings, age, pickup time, order hour) and are handled inside the pipeline.

## CRISP-ML(Q) Steps
### 1. Business & Data Understanding
- Defined the delivery ETA problem as a regression task.
- Inspected the data using info, describe, shape and head.
- Ran correlation analysis, KDE plots and box plots.
- **Quality assurance:** checked duplicates, missing values, distributions and outliers.

### 2. Data Preparation
- Dropped rider ID, latitude/longitude columns, order date and city name.
- Imputed missing values, scaled numeric features and encoded categorical features.
- Split the data 80/20 with a fixed random seed.
- **Quality assurance:** all steps sit inside a Pipeline, so test data never leaks into training.

### 3. Modelling
- Built six models: Linear Regression, KNN, Decision Tree, SVR, Random Forest and XGBoost.
- Selected XGBoost and tuned it with Optuna.
- **Quality assurance:** same data split for all models, bias–variance check and cross-validation.

### 4. Evaluation
- Measured R², MAE, MSE and RMSE on unseen data.
- Compared default and tuned XGBoost (R² improved from 0.819 to 0.831).
- **Quality assurance:** 5-fold cross-validation gave mean R² 0.812 with standard deviation 0.004.

### 5. Deployment
- Saved the full pipeline (preprocessing + tuned XGBoost) as a pickle file for a Streamlit app.
- **Quality assurance:** preprocessing is bundled inside the model, so identical transformations are applied at prediction time.

### 6. Monitoring & Maintenance
- Not yet implemented.
- Planned: track prediction error, detect data drift and retrain periodically.

## Preprocessing Summary
- **Numeric columns:** age, ratings, vehicle_condition, multiple_deliveries, order_day, order_month, is_weekend, pickup_time_minutes, order_time_hour, distance
  - **Steps:** KNN Imputer → Standard Scaler → Power Transformer
  - **Why:** fills gaps using similar rows, equalises scales and reduces skew
- **Nominal categorical columns:** weather, type_of_vehicle, festival, is_weekend, order_day_of_week, order_time_of_day
  - **Steps:** Most-frequent Imputer → One-Hot Encoder
  - **Why:** these have no natural order, and unseen categories are safely ignored
- **Ordinal categorical columns:** traffic (low < medium < high < jam), city_type (semi-urban < urban < metropolitian)
  - **Steps:** Most-frequent Imputer → Ordinal Encoder with explicit order
  - **Why:** preserves the real order of the categories
- **Dropped columns:** rider_id, restaurant/delivery latitude and longitude, order_date, city_name
  - **Why:** they are unique IDs or their information is already captured by other features (distance, city_type, date parts)

## Model Comparison (20% hold-out test set)
- **Linear Regression:** MSE 38.24, MAE 4.93, RMSE 6.18, R² 0.570
- **K-Nearest Neighbours:** MSE 32.28, MAE 4.45, RMSE 5.68, R² 0.637
- **Decision Tree:** MSE 30.55, MAE 4.17, RMSE 5.53, R² 0.656
- **Support Vector Regressor:** MSE 25.89, MAE 4.02, RMSE 5.09, R² 0.709
- **Random Forest:** MSE 16.12, MAE 3.17, RMSE 4.01, R² 0.819
- **XGBoost (default):** MSE 16.07, MAE 3.18, RMSE 4.01, R² 0.819
- **XGBoost (tuned):** MSE 15.03, MAE 3.08, RMSE 3.88, R² 0.831 ✅ **Best model**
- **Observation:** Linear Regression performed worst, which shows the relationships are non-linear. Tree-based ensembles clearly beat single models.

Linear Regression performed worst, showing the relationships are non-linear. Tree ensembles clearly beat single models.

## Cross-Validation
- **Technique:** 5-fold K-Fold on the training set, with shuffling and a fixed random seed for reproducibility.
- **How it works:** the training data is split into 5 parts; the model trains on 4 and validates on 1, rotating until every part has been the validation set. The full pipeline is refitted in every fold, so preprocessing never sees the validation data.
- **Result (default XGBoost):** fold R² scores between 0.808 and 0.818, **mean 0.812, standard deviation 0.004**.
- **Meaning:** scores are nearly identical across folds and close to the test score, so the model is stable and not dependent on one lucky split.

## Bias–Variance Check
- **Random Forest**
  - Train R² 0.974 vs Test R² 0.819 (gap 0.155)
  - Train RMSE 1.51 vs Test RMSE 4.01
  - **Conclusion:** overfits the training data (high variance)
- **XGBoost**
  - Train R² 0.868 vs Test R² 0.819 (gap 0.049)
  - Train RMSE 3.41 vs Test RMSE 4.01
  - **Conclusion:** generalises better
- **Decision:** both models reach the same test accuracy, so XGBoost was selected for its much smaller train–test gap.

Both models reach the same test accuracy, but Random Forest memorises the training data far more. **XGBoost was chosen** for its much smaller train–test gap.

##  Hyperparameter Tuning
- **Technique:** Optuna with Bayesian optimisation (default TPE sampler), which uses results of earlier trials to pick better settings, making it more efficient than grid or random search.
- **Setup:** 50 trials, maximising R², tuning the complete pipeline.
- **Parameters searched:** number of trees (100–1000), tree depth (3–10), learning rate (0.01–0.3, log scale), row sampling (0.5–1.0), column sampling (0.5–1.0), gamma (log scale, regularisation) and minimum child weight (1–10).
- **Best values found:** 400 trees, depth 10, learning rate ≈ 0.0148, row sampling ≈ 0.78, column sampling ≈ 0.88, gamma ≈ 0.003, minimum child weight 9.
- **Impact:** R² improved from 0.819 to 0.831 and MAE from 3.18 to 3.08 minutes.

##  Limitations & Future Work
**Limitations**
- Tuning was scored on the test set, so the final 0.831 is slightly optimistic.
- Cross-validation was run on the default model, not the tuned one.
- `is_weekend` is processed by both the numeric and nominal transformers; `type_of_order` is unused.
- Only point predictions are produced (no confidence intervals); monitoring is not implemented.

**Future work**
- Tune using cross-validation on training data and keep the test set untouched until the end.
- Add prediction intervals and explainability (SHAP).
- Try LightGBM and CatBoost.
- Add drift monitoring, scheduled retraining and a finished Streamlit app.

## How to Run
1. Clone this repository.
2. Install: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, streamlit, jupyter.
3. Place `swiggy_demographic.csv` in the project folder and update the file path in the notebook.
4. Run the notebook top to bottom; the last cells train the final model and save `swiggy.pkl`.
5. Load `swiggy.pkl` with pickle to predict delivery time on new orders.

## Repository Contents
- **Notebook:** full workflow from EDA to model export.
- **swiggy.pkl:** saved final pipeline.
- **swiggy_demographic.csv:** dataset (if licence permits).
- **README.md:** this document.

## Author
**K Sanjay** · GitHub: https://github.com/sanjaykorrawath96-sys· LinkedIn:https://www.linkedin.com/in/korrawath-sanjay-kumar-711b2b414
