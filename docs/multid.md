# Multi-Dimensional Rating Prediction — Full Modeling Plan

## 1. Project Objective

The goal is to build a multi-dimensional rating prediction system that predicts not only the overall restaurant rating but also interpretable aspect-level scores derived from user reviews.

The system outputs:

- Overall predicted rating  
- Food quality score  
- Service score  
- Atmosphere score  
- Value (price-performance) score  

This transforms a single scalar prediction into an interpretable, decision-oriented model.

---

## 2. Problem Definition

### Input
Restaurant-level data including:
- Structured metadata  
- Aggregated review information  
- Text-derived features (embeddings, sentiment, topics)  

### Output

{
  "overall": 4.2,
  "food_positive_ratio": 0.82,
  "service_positive_ratio": 0.74,
  "atmosphere_positive_ratio": 0.79,
  "value_positive_ratio": 0.65
}

### Validation

- Compare aspect sentiment with overall rating:
  - High overall rating → high positive ratios
- Correlation analysis:
  - corr(food_positive_ratio, overall_rating)
- Manual inspection on sampled restaurants



---

## 3. Data Representation

### 3.1 Structured Features

- Categories (cuisine type)  
- Price range  
- City / state  
- Latitude / longitude  
- Attributes (e.g., good for groups, takeout)  
- Review count  

---

### 3.2 Text-Derived Features

#### Review Embeddings
- Aggregate review text per restaurant  
- Use Transformer encoder to generate embedding  

#### Sentiment Features
- Positive vs negative review ratios  
- Average sentiment score  

#### Topic Features
- Extract topics such as:
  - food  
  - service  
  - atmosphere  
  - value  

---

### 3.3 Feature Combination

Final feature vector:

X = [structured_features, embedding_features, sentiment_features, topic_features]

---

## 4. Target Construction

### 4.1 Overall Rating
- Directly from dataset (e.g., Yelp stars)

---

### 4.2 Aspect-Level Ratings (Weak Supervision)

Since explicit labels are unavailable, construct proxy targets.

#### Method

1. Define keyword groups:
   - food → taste, fresh, portion  
   - service → staff, waiter, service  
   - atmosphere → cozy, noisy, ambiance  
   - value → price, cheap, expensive  

2. For each restaurant:
   - Extract sentences related to each aspect  
   - Compute sentiment score per aspect  

3. Normalize scores to rating scale (1–5)

---

## 5. Model Design

### 5.1 Baseline Models

- Mean predictor  
- Linear Regression / Ridge  

---

### 5.2 Tree-Based Models

- Random Forest  
- Gradient Boosted Trees (XGBoost / LightGBM)  

---

### 5.3 Neural Models

#### MLP (Multi-Layer Perceptron)

Input:
- Combined feature vector  

Output:
- overall  
- food  
- service  
- atmosphere  
- value  

---

### 5.4 Multi-Task Learning

Train model to predict all dimensions jointly.

Loss:

L = L_overall + λ1 * L_food + λ2 * L_service + λ3 * L_atmosphere + λ4 * L_value

Benefits:
- Shared representation  
- Better generalization  
- More interpretable outputs  

---

## 6. Training Strategy

### 6.1 Data Split

- Train / validation / test split by restaurant_id  
- Avoid data leakage  

---

### 6.2 Feature Scaling

- Normalize numerical features  
- Standardize embeddings if needed  

---

### 6.3 Regularization

- L2 regularization  
- Dropout (for neural models)  

---

## 7. Evaluation Metrics

### 7.1 Regression Metrics

Primary:
- MAE (Mean Absolute Error)  

Secondary:
- RMSE  
- R²  

---

### 7.2 Ranking Metrics

- Precision@K  
- NDCG@K  

---

### 7.3 Multi-Dimensional Evaluation

- MAE_overall  
- MAE_food  
- MAE_service  
- MAE_atmosphere  
- MAE_value  

---

### 7.4 Qualitative Evaluation

- Inspect predicted aspect scores  
- Compare with review content  
- Evaluate interpretability  

---

## 8. Model Selection Criteria

Models are evaluated based on:

1. Predictive accuracy  
2. Ranking usefulness  
3. Interpretability  
4. Deployment feasibility  

---

## 9. Recommended Modeling Pipeline

### Step 1
- Mean baseline  
- Ridge regression  

### Step 2
- Random Forest  
- Gradient Boosted Trees  

### Step 3
- Add text embeddings  
- Add sentiment features  

### Step 4
- MLP model  

---

## 10. Final Model Choice

Recommended:

Multi-task MLP  

Reason:
- Supports multi-dimensional output  
- Uses both structured and semantic features  
- Aligns with explainable prediction goals  

---

## 11. Deployment Considerations

- Precompute embeddings offline  
- Use lightweight inference pipeline  
- Ensure fast prediction for UI  

---

## 12. Limitations

- Aspect labels are weakly supervised  
- Bias in review data  
- Uneven geographic coverage  

---

## 13. Future Improvements

- Use LLM for better aspect extraction  
- Personalized rating prediction  
- Temporal dynamics  
- Better text modeling  

---

## 14. Summary

This system extends traditional rating prediction by introducing multi-dimensional, interpretable outputs. By combining structured features and Transformer-based embeddings, the model provides both accurate predictions and meaningful explanations, making it suitable for real-world decision support.