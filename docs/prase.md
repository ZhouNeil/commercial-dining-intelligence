# Restaurant Recommendation System — Transformer-Based Implementation (Plan A)

## 1. Project Goal

Build a semantic, explainable restaurant recommendation system that takes natural language queries such as "cheap sushi near NYU" and returns:

- Top-K recommended restaurants  
- Clear explanations (why recommended)  
- Pros and cons extracted from reviews  

---

## 2. System Pipeline

User Query  
Query Parser  
Structured Constraints  
Semantic Query (Text)  
Transformer Embedding  
FAISS Retrieval (Top-N)  
Filtering (location, price, cuisine)  
Re-ranking (multi-factor scoring)  
Insight Generation (pros, cons, explanation)  
Streamlit UI  

---

## 3. Core Components

### 3.1 Query Parser

Objective  
Convert user input into structured constraints.

Example  

Input  
cheap sushi near NYU  

Output  

{
  "cuisine": "sushi",
  "budget": "cheap",
  "location": "NYU",
  "radius_km": 2,
  "vibe_terms": ["sushi"]
}

Implementation  
- Rule-based keyword extraction  
- Predefined dictionaries:
  - cuisine  
  - budget  
  - location  
  - vibe keywords  

---

### 3.2 Restaurant Profile Construction

Objective  
Create a unified semantic representation for each restaurant.

Template  

Restaurant: Joe's Sushi  
Category: Sushi, Japanese  
City: New York  
Price: $$  
Attributes: casual dining, good for dinner  
Positive themes: fresh fish, friendly staff, good value  
Negative themes: long wait, small seating area  

Data Sources  
- Business metadata  
- Aggregated review insights  

---

### 3.3 Review Processing

Method  

Split reviews into:
- Positive reviews (stars greater or equal to 4)  
- Negative reviews (stars less or equal to 2)  

Extract:
- Top positive phrases  
- Top negative phrases  

Example  

Positive: fresh fish, great service, cozy atmosphere  
Negative: long wait, noisy environment  

---

### 3.4 Transformer Embedding

Model  
sentence-transformers all-MiniLM-L6-v2  

Process  
- Input: restaurant profile text  
- Output: dense vector representation  

Storage  
- restaurant_embeddings.npy  
- restaurant_profiles.csv  

---

### 3.5 Vector Retrieval

Offline  
- Build FAISS index from restaurant embeddings  

Online  
- Convert query into embedding  
- Retrieve Top-N similar restaurants  

---

### 3.6 Query Understanding

Split query into:

Structured part  
- budget  
- location  
- cuisine  

Semantic part  
- sushi restaurant  
- quiet cafe for studying  

Only the semantic part is used for embedding.

---

### 3.7 Filtering

Apply constraints:

- City or state  
- Distance (using latitude and longitude)  
- Price range  
- Cuisine category  

Goal  
Remove irrelevant candidates before ranking  

---

### 3.8 Re-ranking

Final score is computed as:

final_score =  
    0.45 times semantic_similarity  
  + 0.20 times rating_score  
  + 0.15 times popularity_score  
  + 0.10 times price_match_score  
  + 0.10 times distance_score  

Components  

- semantic_similarity from embedding similarity  
- rating_score from restaurant rating  
- popularity_score from review count  
- price_match_score based on budget match  
- distance_score based on proximity  

---

### 3.9 Insight Generation

Each restaurant displays:

Why Recommended  
- Matches query intent  
- Close to user location  
- High rating  

Pros  
- Extracted from positive phrases  

Cons  
- Extracted from negative phrases  

---

### 3.10 Streamlit UI

Layout  

Top section  
- Search input  

Sidebar  
- Filters (price, distance, city)  
- Top-K selection  

Main section  
- Restaurant cards:
  - Name  
  - Rating  
  - Distance  
  - Price  
  - Explanation  
  - Pros  
  - Cons  

---

## 4. Data Pipeline

Input  
- Yelp business dataset  
- Yelp review dataset  

Processing Steps  
1. Clean data  
2. Aggregate reviews by restaurant  
3. Extract key phrases  
4. Build restaurant profiles  
5. Generate embeddings  

---

## 5. Tech Stack

Core  
- Python  
- pandas  
- numpy  

Machine Learning  
- sentence-transformers  
- faiss  
- scikit-learn  

Natural Language Processing  
- KeyBERT  
- nltk or spacy  

Frontend  
- Streamlit  

---

## 6. Project Structure

project/

data/
- businesses.csv  
- reviews.csv  
- restaurant_profiles.csv  
- restaurant_embeddings.npy  
- faiss.index  

src/
- preprocess.py  
- build_profiles.py  
- embed_restaurants.py  
- query_parser.py  
- retrieve.py  
- filter_rank.py  
- insight.py  
- app.py  

requirements.txt  
README.md  

---

## 7. Development Phases

Phase 1 Retrieval  
- Build profiles  
- Generate embeddings  
- Implement FAISS retrieval  

Phase 2 Understanding and Ranking  
- Build query parser  
- Add filtering  
- Implement re-ranking  

Phase 3 Insight and UI  
- Extract pros and cons  
- Generate explanations  
- Improve Streamlit interface  

---

## 8. Evaluation

Quantitative  
- Precision at K  
- Retrieval relevance  

Qualitative  
- User satisfaction  
- Interpretability of results  

---

## 9. Key Design Principle

Use Transformer-based embeddings for semantic understanding, combined with structured constraints and multi-factor ranking, to produce meaningful and explainable recommendations.