# Restaurant Recommendation System — Development Document

## 1. Project Overview

This project aims to build a **user-facing restaurant recommendation and analysis system** that helps users make dining decisions based on natural language queries (e.g., "cheap sushi near NYU").

The system integrates:
- Query understanding
- Retrieval (embedding-based + optional keyword-based)
- Multi-factor ranking
- Insight extraction from reviews

Unlike traditional search systems, this application not only returns recommendations but also explains *why* each result is relevant.

---

## 2. System Architecture

```
User Query
→ Query Parser
→ Structured Constraints
   - cuisine
   - budget
   - location
→ Candidate Retrieval
   - dense embedding retrieval
   - optional sparse retrieval
→ Filtering
   - state/city
   - distance
   - price
→ Re-ranking
   - semantic similarity
   - rating
   - popularity
   - distance
→ Insight Generation
   - pros
   - cons
   - why recommended
→ Streamlit UI
```

---

## 3. Module Breakdown

### 3.1 Query Parser

**Objective**  
Convert natural language input into structured constraints.

**Input**
- Raw user query (string)

**Output**
```json
{
  "cuisine": "sushi",
  "budget": "cheap",
  "location": "NYU",
  "radius_km": 2
}
```

**Implementation**
- Rule-based keyword matching (initial version)
- Optional: LLM-based parsing (future improvement)

**Tools**
- Python (regex, dictionaries)

---

### 3.2 Structured Constraints

**Components**
- Cuisine (e.g., sushi, italian)
- Budget (cheap, moderate, expensive)
- Location (coordinates or predefined mapping)
- Radius (distance threshold)

**Notes**
- Budget maps to price range (`$`, `$$`, `$$$`)
- Location converts to latitude/longitude

---

### 3.3 Candidate Retrieval

**Objective**  
Retrieve a pool of relevant restaurants.

**Methods**

**Dense Retrieval (Primary)**
- Convert restaurant data into embeddings
- Convert query into embedding
- Compute similarity (cosine similarity)

**Sparse Retrieval (Optional)**
- TF-IDF or BM25 for keyword matching

**Output**
- Top-N candidate restaurants (e.g., N = 100)

**Tools**
- sentence-transformers  
- faiss or sklearn  
- rank-bm25 (optional)

---

### 3.4 Filtering

**Objective**  
Apply hard constraints before ranking.

**Filters**
- State / City  
- Distance (via coordinates)  
- Price range  
- Cuisine category  

**Implementation**
- pandas filtering  
- Haversine distance calculation  

---

### 3.5 Re-ranking

**Objective**  
Rank candidates based on multiple factors.

**Scoring Function**

```
score = w1 * semantic 
      + w2 * rating 
      + w3 * price_match 
      + w4 * distance_score 
      + w5 * popularity
```

**Features**
- Semantic similarity (embedding)  
- Yelp rating  
- Review count (popularity)  
- Distance score  
- Price match score  

**Implementation**
- Weighted scoring (rule-based)

---

### 3.6 Insight Generation

**Objective**  
Provide explanations for recommendations.

**Outputs**
- Pros (positive aspects)  
- Cons (negative aspects)  
- Why recommended  

**Methods**

**Basic**
- Keyword extraction  
- Frequency analysis  

**Advanced**
- Embedding + clustering  
- Sentiment grouping  

**Tools**
- KeyBERT  
- sklearn  
- Optional: LLM for summarization  

---

### 3.7 Streamlit UI

**Objective**  
Provide an interactive interface.

**Features**
- Search input box  
- Top-K recommendations  
- Restaurant cards:
  - Name  
  - Rating  
  - Distance  
  - Price  
- Insight section (pros / cons / explanation)  

**Optional Enhancements**
- Map visualization (pydeck, folium)  
- Filters (price, cuisine)  

---

## 4. Data Pipeline

**Input Data**
- Yelp dataset:
  - Business metadata  
  - Reviews  

**Preprocessing**
- Clean text  
- Aggregate reviews per restaurant  
- Generate embeddings  
- Store structured attributes  

---

## 5. Tech Stack Summary

**Core**
- Python  
- pandas, numpy  

**ML / Retrieval**
- sentence-transformers  
- faiss / sklearn  
- TF-IDF / BM25 (optional)  

**NLP**
- KeyBERT  
- nltk / spacy  

**Frontend**
- Streamlit  

**Optional**
- LLM (OpenAI / local)  
- Map visualization tools  

---

## 6. Evaluation Plan

**Quantitative**
- Precision@K  
- Similarity scores  

**Qualitative**
- User experience testing  
- Interpretability of recommendations  

---

## 7. Future Improvements

- Personalized recommendation (user embeddings)  
- Learning-to-rank model  
- LLM-based query parsing  
- Real-time API integration  
- Advanced visualization  

---

## 8. Conclusion

This system transforms traditional keyword search into a **semantic, explainable recommendation engine**. By combining embedding-based retrieval, structured filtering, and insight generation, it provides both actionable recommendations and interpretable reasoning aligned with real-world decision-making.