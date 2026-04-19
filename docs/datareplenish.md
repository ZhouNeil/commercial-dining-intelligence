# Yelp API Data Collection Plan (NY / NJ / PA)

## 1. Goal

The main problem is not just that the NYC dataset is small, but that it has:

- incomplete coverage
- uneven geographic distribution
- missing long-tail restaurants
- weak semantic information

The goal of this plan is to:

1. Expand coverage to NY, NJ, and PA
2. Build a unified restaurant master dataset
3. Support future hybrid retrieval (embedding + keyword)
4. Ensure the pipeline is reproducible, debuggable, and scalable

---

## 2. High-Level Strategy

We do NOT try to scrape everything at once.

Instead, we use:

- geographic partitioning
- query decomposition
- deduplication
- validation

Pipeline:

User-defined regions  
→ Split into smaller areas  
→ Query Yelp API multiple times  
→ Merge results  
→ Deduplicate  
→ Validate  
→ Build master dataset  

---

## 3. Region Design

Focus on tri-state area:

### New York
- Manhattan
- Brooklyn
- Queens
- Bronx
- Staten Island

### New Jersey
- Jersey City
- Hoboken
- Newark
- Fort Lee
- Edison

### Pennsylvania
- Philadelphia
- Pittsburgh (optional)
- Allentown (optional)

---

## 4. Geographic Partitioning

Instead of querying entire cities, divide into smaller regions.

Two approaches:

### Option A (recommended)
Use latitude and longitude with radius.

Each query:
- center point
- radius (e.g. 2000 meters)

Create a grid:
- NYC: around 20 to 40 points
- NJ: around 10 to 20 points
- PA: around 10 to 20 points

---

## 5. Query Decomposition

Because Yelp API limits results per query, we must split queries.

### Categories

Use multiple cuisine categories:

- restaurants (general)
- chinese
- japanese
- korean
- italian
- mexican
- thai
- indian
- american
- pizza
- fastfood

### Price Levels

Split into:

- 1,2 (cheap and moderate)
- 3,4 (expensive)

---

## 6. API Execution Logic

For each region:

For each grid point  
For each category  
For each price group  
For each offset (0, 50, 100, 150, 200)  

Call Yelp API

Collect results

---

## 7. Deduplication

Yelp provides business_id.

Use it as the primary key.

If duplicate appears:
- keep only one entry

Later, when merging with other sources:
- use name + distance to match

---

## 8. Data Schema

Final dataset should have one row per restaurant:

- restaurant_id
- name
- address
- city
- state
- zip_code
- latitude
- longitude
- categories
- price_tier
- stars
- review_count
- phone
- source

---

## 9. Data Validation

After collection, verify the dataset.

### Coverage Check
- number of restaurants per region
- ensure all major areas are represented

### Distribution Check
- categories are balanced
- not dominated by one cuisine

### Duplicate Check
- ensure unique restaurant_id
- no duplicated names at same location

### Missing Data Check
- percentage of missing fields
- price, category, location completeness

---

## 10. Execution Verification

You know the pipeline is working correctly if:

1. Total restaurant count increases significantly
2. Each region has reasonable coverage
3. No large number of duplicates
4. Queries from different locations return results
5. Different cuisines are represented

---

## 11. Output

Produce one main file:

restaurant_master.csv

This file should be the ONLY input for:
- retrieval
- ranking
- embedding generation

---

## 12. Integration with Recommender System

After dataset is ready:

1. Build restaurant profiles
2. Generate text features
3. Generate embeddings
4. Plug into hybrid retrieval

Future system:

- 30 percent keyword matching
- 70 percent embedding similarity

---

## 13. Key Principle

Do NOT try to collect every restaurant.

Focus on:

- sufficient coverage
- balanced distribution
- rich information per restaurant

A good dataset is not the largest one,
but the one that supports strong retrieval and ranking.

---

## 14. Runnable script (repo)

Implementation: `scripts/yelp_replenish.py` (Yelp Fusion v3 `/businesses/search`).

Prerequisites:

- `pip install -r requirements.txt` (includes `requests`)
- [Yelp Fusion](https://docs.developer.yelp.com/docs/fusion-intro) API key in env: `YELP_API_KEY`

Examples:

```bash
# 预估任务量（全量网格约 1100+ 组合 × 分页，慎用配额）
python scripts/yelp_replenish.py --dry-run

# 少量代表城市 + 菜系 + 价位组（推荐先跑通）
export YELP_API_KEY=your_key_here
python scripts/yelp_replenish.py --small-grid --max-offset 50 --output data/cleaned/restaurant_master.csv

# 冒烟：仅第一个组合、单页
python scripts/yelp_replenish.py --smoke --max-requests 1

# 断点续跑（去重合并到输出）
python scripts/yelp_replenish.py --small-grid --resume data/cleaned/restaurant_master.csv --output data/cleaned/restaurant_master.csv
```

Output schema matches §8 (`restaurant_id`, …, `source`), plus `yelp_url` and `collection_region` for debugging.
