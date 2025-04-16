Final Project Report:
Weather Forecasting using LSTM and GRU Models

Introduction

This project focuses on building deep learning models to predict weather conditions, specifically temperature, using historical weather data. The goal is to compare the performance of two popular recurrent neural network architectures: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU).
Key Objectives:
1.Data Preprocessing: 
Clean,remove nulls, normalize, and transform raw weather data into a suitable format for deep learning models.
2. Feature Engineering:
 Extract meaningful temporal features and encode categorical variables.
3. Model Development:
 Implement and train LSTM and GRU models for time-series forecasting.
4. Evaluation:
Compare model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
5. Visualization:
Generate plots to analyze model predictions and residuals.



2. Dataset Overview
The dataset used is( weatherHistory.csv), containing historical weather records with the following key features:

Features:
- Numerical Features:
  - `Temperature (C)` (Target Variable)  
  - `Humidity`, `Wind Speed (km/h)`, `Pressure (millibars)`, etc.  
- Categorical Features:
  - `Summary` (Weather description)  
  - `Precip Type` (Rain, snow, etc.)  
- Temporal Feature:
  - `Formatted Date` (Timestamp in UTC)  

Initial Data Exploration:
- Shape:
 Original dataset had 96,453 entries.
- Missing Values:
 `Precip Type` had missing values, which were filled using forward fill (`ffill`).
- Outliers:
 Detected and removed using the IQR method.
- Dropped Irrelevant Features:
  - `Daily Summary` (Redundant with `Summary`)  
  - `Loud Cover` (Constant value, no predictive power)  
  - `Apparent Temperature (C)` (Highly correlated with `Temperature (C)`)  

---
3-Data Preprocessing & Feature Engineering:
3.1 Handling Missing Values & Outliers
- Missing Values:
 Filled `Precip Type` using forward fill.
- Outlier Removal:
Applied IQR-based filtering to remove extreme values in numerical features.

3.2 Feature Engineering
Time-Based Features:
- Extracted `Hour`, `Day of Week`, and `Month` from the timestamp.
- Added a rolling mean (`Rolling Temp`) to capture short-term trends.
- Cyclical Encoding: 
Used sine and cosine transformations to encode periodic patterns:
  - `Hour_sin`, `Hour_cos`  
  - `Day_sin`, `Day_cos`  
  - `Month_sin`, `Month_cos`  

Categorical Encoding:
- Applied **One-Hot Encoding** to `Summary` and `Precip Type`.

Normalization:
- Scaled numerical features using `MinMaxScaler` (range: `[0, 1]`).



4. Model Architecture
4.1 LSTM Model
- Architecture:  
  - Bidirectional LSTM:
 (2 layers, 128 hidden units)  
  - Dropout (0.3)   :  for regularization  
  - Batch Normalization :  for stable training  
  - ReLU Activation : for non-linearity  
  - Final Linear Layer: for regression output  

4.2 GRU Model
- Architecture:
  - Bidirectional GRU :(2 layers, 128 hidden units)  
  - Dropout (0.3) : for regularization  
  - Batch Normalization: for stable training  
  - ReLU Activation: for non-linearity  
  - Final Linear Layer: for regression output  

Training Details:
- Optimizer: AdamW (learning rate = 0.0005, weight decay = 1e-4)  
- Loss Function:Mean Squared Error (MSE)  
- Batch Size:32  
- Epochs:30  
- Train-Validation-Test Split:  
  - Train: 70%  
  - Validation:15%  
  - Test: 15%  



5. Model Evaluation
5.1 Performance Metrics
| Model  | MAE  | MSE  | R² Score |
|--------|------|------|----------|
| LSTM   | 0.023 | 0.001 | 0.998 |
| GRU    | 0.024 | 0.001 | 0.998 |

5.2 Key Observations:
1. Both models performed exceptionally well, with near-perfect R² scores (~0.998).
2. LSTM slightly outperformed GRU in terms of MAE and MSE.
3. Residual Analysis:
   - Residuals were randomly distributed around zero, indicating no systematic bias.  
   - No evident patterns in errors, suggesting good model fit. 

5.3 Prediction Visualization
- Actual vs. Predicted Plots:
  - Both models closely followed the true temperature trends.  
  - Minor deviations observed in highly fluctuating regions.  
- Zoomed-in Analysis (First 100 samples):
  - LSTM captured short-term variations slightly better than GRU.  



6.Conclusion & Future Work
6.1 Findings
- LSTM and GRU both excel in weather forecasting due to their ability to capture temporal dependencies.
- LSTM was marginally better, possibly due to its more complex gating mechanism.
- Feature engineering (cyclical encoding, rolling mean)significantly improved model performance.

6.2 Limitations
- Short sequence length (24 steps):Longer sequences might improve predictions.
- Static test set: Real-world deployment requires online learning for adaptability.

6.3 Future Improvements
1. Hyperparameter Tuning: Grid search for optimal layers, hidden units, and dropout.
2. Attention Mechanism:Enhance model focus on critical time steps.
3. Multivariate Forecasting:Predict multiple weather variables simultaneously.
4.Deployment:Integrate with real-time weather APIs for live predictions.

---

7. Final Thoughts
This project successfully demonstrated the application of deep learning in weather forecasting. The high accuracy of both LSTM and GRU models suggests that recurrent architectures are well-suited for time-series weather data. Future enhancements could make the system even more robust for real-world applications.  

Dataset Source: https://www.kaggle.com/datasets/muthuj7/weather-dataset
