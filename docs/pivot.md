# Yelp Project Redesign Directions

## Direction 1: Restaurant Recommendation System

### Problem
Users often struggle to decide where to eat given multiple constraints such as location, budget, and cuisine preferences.

### Method
- Use embedding models to represent restaurant reviews and metadata  
- Apply similarity search / nearest neighbor retrieval  
- Rank restaurants based on relevance to user input  

### Can You Do It With Your Current Data?
Yes—you can. The implementation difficulty is low-to-medium, because you already have:
- `data/cleaned/business_dining.csv`: restaurant metadata such as `city/state/latitude/longitude/categories/stars/review_count/is_open`, useful for candidate filtering and re-ranking.
- `data/cleaned/review_dining.csv`: review text (`text`), `user_id`, `business_id`, plus signals like `stars/useful/funny/cool/date`, useful for building restaurant vectors (and optionally user preference vectors).
- `data/cleaned/tip_dining.csv / checkin_dining.csv` (optional): additional behavioral signals; tip text can be embedded too or used to improve weighting.

Limitations:
- You do not have a separate `user_dining` table, but `review_dining.csv` already contains `user_id`, so personalized recommendations are still feasible using review history.
- Your recommendation can be framed as “similar-restaurant retrieval + personalized ranking” without requiring additional labeled data.

### Data Needed (Use These Files)
1. `data/cleaned/business_dining.csv`
   - Required fields: `business_id, name, city, state, latitude, longitude, stars, review_count, is_open, categories`
   - Optional fields: `attributes` (e.g., `RestaurantsPriceRange2`) and `hours`
2. `data/cleaned/review_dining.csv`
   - Required fields: `business_id, user_id, text, stars, useful, date`
   - Optional fields: `funny, cool` (can be used as helpfulness/weighting signals)
3. （可选）`data/cleaned/tip_dining.csv`
   - If used: embed `text` as additional signals or incorporate tips into aggregation weights

### Recommended Technical Stack
- Python (data processing + training + offline index construction)
- `pandas`, `numpy`
- `sentence-transformers` (local embeddings; can be replaced with OpenAI embeddings)
- Vector index / nearest neighbors:
  - Recommended: `faiss-cpu` (fast, suitable for offline indexing)
  - Alternative: `scikit-learn` `NearestNeighbors` (simpler, slower)
- `streamlit` (a demo web app)

Your current `requirements.txt` already includes `pandas/numpy/scikit-learn/streamlit`. You will likely need:
- `sentence-transformers`
- (recommended) `faiss-cpu`

### Algorithm Requirements (What to Implement)
1. Data cleaning & candidate set
   - Use your existing dining filtering logic to ensure the candidate set contains only restaurant businesses.
   - Text processing: clean `review_dining.text` (drop empty text, truncate to model max length) and limit to up to N reviews per business (to keep offline embedding manageable).
2. Restaurant Embedding
   - For each `business_id`, embed its review `text`.
   - Aggregation (implement at least one):
     - Mean: `restaurant_vec = mean(review_vecs)`
     - Weighted mean (recommended): weights from `stars` or `useful+funny+cool`, plus time decay (more recent reviews get higher weight).
3. User Preference Vector (optional but recommended)
   - For each `user_id`, aggregate a preference vector from their history.
   - Common approaches:
     - Aggregate vectors from the user’s review texts directly.
     - Aggregate vectors of the restaurants the user reviewed (often more stable).
4. Similarity Search / Retrieval
   - Inputs:
     - A) User vector (personalized): `topK = ANN(user_vec)`
     - B) User keywords (non-personalized / semi-personalized): turn keywords into a query embedding, then retrieve.
   - If you need geo/budget constraints:
     - Filter candidates using `business_dining` fields (`city/state/categories/attributes`).
     - Run vector retrieval on the candidate set (or post-filter + re-rank).
5. Ranking / Re-ranking
   - Base score: vector similarity (cosine/dot).
   - Optional priors to improve quality:
     - Priors: `stars`, `review_count`, `is_open`
     - Geo distance decay using `latitude/longitude` and user location (if available).
   - Example final score: `score = alpha * sim + beta * stars_prior + gamma * geo_decay`
6. Offline / online workflow
   - Offline: build restaurant vectors + build/save a FAISS index.
   - Online (Streamlit): load index -> user input -> embed(query/user) -> retrieve -> display Top-K restaurants.

### Evaluation (How to Measure Accuracy)
Since you do not have explicit “purchase / next time they will visit” labels, use a weakly supervised, time-based split:
- Split per `user_id` by `review_dining.date`: build a user vector from past reviews.
- Hold out the most recent (or last N) reviewed `business_id` as ground truth.
- Metrics:
  - `Recall@K`: whether ground truth appears in Top-K
  - `NDCG@K`: rewards correct ranking positions

Also compare against baselines:
- Popularity baseline (sort by `review_count` or `stars`)
- Random baseline

### Web App Demo (Streamlit)
- Input:
  - `user_id` (personalized recommendations), or
  - `city/state` + `keywords` (e.g., “pizza/coffee/bubble tea”)
  - Optional: budget (infer from `attributes.RestaurantsPriceRange2`)
- Output:
  - Top-K restaurants: show `name/city/state/stars/review_count` plus a short explanation (high-level reason from review-text similarity and/or category matching)

### Real-world Value
- Helps users make dining decisions  
- Mimics real-world applications like Yelp or Google Maps  
- Provides clear user-facing utility  

### Key Strength
- Strong alignment with course expectations  
- Clear user scenario  
- Easy to demonstrate in a web app  


---

## Direction 2: Review Insight Analyzer

### Problem
Users cannot easily understand why a restaurant has a certain rating by reading hundreds of reviews.

### Method
- Use text embeddings on reviews  
- Apply clustering or classification  
- Extract:
  - common positive themes  
  - common negative themes  

### Real-world Value
- Provides interpretable insights behind ratings  
- Helps users quickly evaluate restaurants  
- Useful for both customers and business owners  

### Key Strength
- More technically focused (NLP-heavy)  
- Strong for algorithm implementation scoring  


---

## Direction 3: Restaurant Success Predictor

### Problem
Entrepreneurs want to estimate how successful a restaurant might be before opening it.

### Method
- Use regression models to predict rating  
- Input features:
  - location  
  - price range  
  - category  
- Optionally incorporate similarity to existing restaurants  

### Real-world Value
- Supports business decision-making  
- Turns prediction into actionable insights  

### Key Strength
- Minimal changes from current Yelp rating prediction project  
- Easy transition by reframing the problem  


---

## Summary

This project should focus on building a user-facing application that answers a real-world question, rather than simply training a predictive model.