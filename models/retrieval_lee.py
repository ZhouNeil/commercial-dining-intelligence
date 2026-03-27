import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class TouristRetrieval:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['categories'] = self.df['categories'].fillna('').str.lower()
        self.df = self.df[self.df['is_open'] == 1].copy()
        self.vectorizer = CountVectorizer(stop_words=['food', 'restaurants'])
        self.vectorizer.fit(self.df['categories'])
        
    def get_recommendations(self, cuisine_query, min_stars=4.0, max_price=None, top_k=5):
        # rule based filters
        query_str = f"stars >= {min_stars}"
        
        if max_price is not None:
            query_str += f" and attr_restaurantspricerange2 <= {max_price} and attr_restaurantspricerange2 > 0"
            
        filtered_df = self.df.query(query_str).copy()
        
        if filtered_df.empty:
            return "No matches found for those filters."

        subset_matrix = self.vectorizer.transform(filtered_df['categories'])
        query_vec = self.vectorizer.transform([cuisine_query.lower()])
        
        # cosine similarity
        scores = cosine_similarity(subset_matrix, query_vec).flatten()
        filtered_df['similarity'] = scores

        # sort by score then stars
        results = filtered_df.sort_values(by=['similarity', 'stars'], ascending=False)
        
        return results.head(top_k)[['similarity', 'name', 'stars', 'attr_restaurantspricerange2', 'categories']]

path = Path("../data/output_philly.csv")
engine = TouristRetrieval(path)

res1 = engine.get_recommendations("Japanese Ramen Sushi", min_stars=4.0, max_price=2)
print(res1)

res2 = engine.get_recommendations("Mediterranean", min_stars=4.5, max_price=None)
print(res2)
