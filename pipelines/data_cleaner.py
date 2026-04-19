import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import MultiLabelBinarizer
from spatial_feature_engineer import SpatialFeatureEngineer

class YelpDataProcessor:
    def __init__(self, business_path):
        """
        Initialize the processor with the path to the business dataset CSV.
        """
        self.business_path = business_path
        self.df = None
        
    def load_data(self):
        print(f"Loading data from {self.business_path}...")
        self.df = pd.read_csv(self.business_path)
        return self.df
        
    def clean_missing(self):
        print("Cleaning missing required rows...")
        self.df = self.df.dropna(subset=['address', 'postal_code', 'attributes', 'categories'])
        
    def filter_data(self, city=None, category_keywords=None):
        print("Filtering data...")
        if city:
            self.df = self.df[self.df['city'].str.lower() == city.lower()]
            
        if category_keywords:
            def has_keyword(cats):
                if not isinstance(cats, str): 
                    return False
                return any(kw.lower() in cats.lower() for kw in category_keywords)
            
            self.df = self.df[self.df['categories'].apply(has_keyword)]
            
        self.df = self.df.copy()
            
    def process_hours(self):
        print("Processing hours...")
        def extract_hours(hours_dict):
            if not isinstance(hours_dict, dict) or len(hours_dict) == 0:
                return pd.Series([0, 0, 0])
            
            is_open_morning = 0
            is_open_latenight = 0
            open_on_weekends = 0

            for day, hours in hours_dict.items():
                try:
                    freq = hours.split('-')
                    open_hour = int(freq[0].split(':')[0])
                    close_hour = int(freq[1].split(':')[0])
                    
                    if open_hour < 10: 
                        is_open_morning = 1
                    if close_hour >= 22: 
                        is_open_latenight = 1
                    if day in ['Saturday', 'Sunday']: 
                        open_on_weekends = 1
                except:
                    continue
            return pd.Series([is_open_morning, is_open_latenight, open_on_weekends])

        self.df['hours'] = self.df['hours'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        hours_features = self.df['hours'].apply(extract_hours)
        hours_features.columns = ['time_is_open_morning', 'time_is_open_latenight', 'time_open_on_weekends']
        
        self.df = pd.concat([self.df, hours_features], axis=1)
        self.df[['time_is_open_morning', 'time_is_open_latenight', 'time_open_on_weekends']] = self.df[['time_is_open_morning', 'time_is_open_latenight', 'time_open_on_weekends']].fillna(0).astype(int)

    def process_categories(self):
        print("Processing categories...")
        self.df['categories'] = self.df['categories'].apply(
            lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else []
        )
        
        mlb = MultiLabelBinarizer()
        categories_df = pd.DataFrame(
            mlb.fit_transform(self.df['categories']),
            columns=mlb.classes_,
            index=self.df.index
        )
        
        # Add prefix and clean category names
        categories_df.columns = ['cat_' + str(col).lower().replace(' ', '_') for col in categories_df.columns]
        self.df = pd.concat([self.df, categories_df], axis=1)

    def process_attributes(self):
        print("Processing and flattening attributes (and dropping useless parent columns)...")
        def safe_parse(x):
            if isinstance(x, dict): 
                return x
            if isinstance(x, str):
                try: 
                    return ast.literal_eval(x)
                except: 
                    return {}
            return {}

        def clean_value(val):
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('{') and val.endswith('}'):
                    try: 
                        return ast.literal_eval(val)
                    except: 
                        return {}
                val = val.replace("u'", "").replace("'", "")
                if val.lower() == 'true': return 1
                if val.lower() == 'false': return 0
                if val.isdigit(): return int(val)
                return val
            return val

        parent_cols_to_drop = set()

        def flatten(attr_dict):
            result = {}
            if not isinstance(attr_dict, dict): 
                return result
                
            for key, val in attr_dict.items():
                val = clean_value(val)
                
                # If nested dictionary, flatten it and mark parent for tracking
                if isinstance(val, dict):
                    parent_cols_to_drop.add(f"attr_{key}")
                    for sub_key, sub_val in val.items():
                        sub_val = clean_value(sub_val)
                        result[f"attr_{key}_{sub_key}"] = 1 if sub_val in [True, 1] else 0
                else:
                    col_name = f"attr_{key}"
                    if val in [True, 1]: 
                        result[col_name] = 1
                    elif val in [False, 0]: 
                        result[col_name] = 0
                    else: 
                        result[col_name] = val
            return result

        self.df['attributes'] = self.df['attributes'].apply(safe_parse)
        attr_df = self.df['attributes'].apply(flatten).apply(pd.Series)

        # Clean column names
        attr_df.columns = [re.sub(r'[^a-z0-9]+', '_', str(col).lower()).strip('_') for col in attr_df.columns]
        
        # Clean parent tracking column names so they match
        cleaned_parents = [re.sub(r'[^a-z0-9]+', '_', str(col).lower()).strip('_') for col in parent_cols_to_drop]

        attr_df = attr_df.fillna(0)
        
        # Drop redundant parent columns that were flattened
        for col in cleaned_parents:
            if col in attr_df.columns:
                attr_df = attr_df.drop(columns=[col])

        self.df = pd.concat([self.df, attr_df], axis=1)
        self.df = self.df.drop(columns=['attributes'])

    def clean_specific_attributes(self):
        print("Cleaning up specific parsed attribute columns...")
        
        # 1. Price Range
        if 'attr_restaurantspricerange2' in self.df.columns:
            self.df['attr_restaurantspricerange2'] = (
                pd.to_numeric(self.df['attr_restaurantspricerange2'], errors='coerce')
                .fillna(0)
                .astype(int)
            )

        # 2. Extract strictly categorical attributes to one-hot encode
        categorical_cols = [
            'attr_wifi', 'attr_alcohol', 'attr_restaurantsattire', 'attr_noiselevel',
            'attr_smoking', 'attr_byobcorkage', 'attr_agesallowed'
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                # Replace 0 with 'unknown' for object categories
                self.df[col] = self.df[col].replace(0, 'unknown')
                self.df[col] = self.df[col].fillna('unknown')

        existing_cat_cols = [col for col in categorical_cols if col in self.df.columns]
        if existing_cat_cols:
            self.df = pd.get_dummies(self.df, columns=existing_cat_cols, prefix=existing_cat_cols)
            
            # Convert the new get_dummies boolean columns to int
            dummy_cols = [col for col in self.df.columns if any(col.startswith(p + "_") for p in existing_cat_cols)]
            self.df[dummy_cols] = self.df[dummy_cols].astype(int)

        # 3. Handle binary columns 
        # Identify columns that are naturally binary + None, or strictly listed binary columns
        binary_candidates = [
            'attr_restaurantsdelivery', 'attr_outdoorseating', 'attr_businessacceptscreditcards',
            'attr_bikeparking', 'attr_restaurantstakeout', 'attr_caters', 
            'attr_restaurantsreservations', 'attr_restaurantsgoodforgroups', 'attr_hastv',
            'attr_goodforkids', 'attr_dogsallowed', 'attr_wheelchairaccessible',
            'attr_drivethru', 'attr_corkage'
        ]
        
        for col in [c for c in self.df.columns if c.startswith('attr_')]:
            # if logically binary
            is_binary = self.df[col].dropna().isin([0, 1, 'None', 'True', 'False', True, False]).all()
            if is_binary or col in binary_candidates:
                self.df[col] = self.df[col].replace('None', 0).replace('True', 1).replace('False', 0)
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        # 4. Filter remaining strictly 'None' string value rows
        cols_with_none = [col for col in self.df.columns if self.df[col].astype(str).eq('None').any()]
        if cols_with_none:
            self.df = self.df.drop(columns=cols_with_none)

    def process(self, city=None, category_keywords=None):
        """
        Runs the entire data cleaning pipeline.
        :param city: Filter dataset to a specific city (str), optional
        :param category_keywords: Filter dataset by category keywords (list of str), optional
        """
        self.load_data()
        self.clean_missing()
        self.filter_data(city=city, category_keywords=category_keywords)
        
        self.process_hours()
        self.process_attributes()
        self.clean_specific_attributes()
        
        self.process_categories()
        return self.df

    


if __name__ == "__main__":
    business_file = "/Users/jackbai/Documents/ML_project/yelp_dataset_csv/yelp_academic_dataset_business.csv"
    
    try:
        # Create an instance
        processor = YelpDataProcessor(business_file)
        
        # Example 1: Run specifically for Philly Restaurants (as per notebook)
        processor.process(city="Philadelphia", category_keywords=["Restaurants", "Food"])
        
        spatial_engineer = SpatialFeatureEngineer(processor.df)
        train_spatial, test_spatial = spatial_engineer.split_and_engineer_spatial_features()
        train_spatial.to_csv("../../train_spatial.csv", index=False)
        test_spatial.to_csv("../../test_spatial.csv", index=False)
        print(f"Data processed and split for Philadelphia successfully! Train: {train_spatial.shape}, Test: {test_spatial.shape}")
        
        # Example 2: How to query everything (uncomment to process the whole dataset)
        # all_restaurants_df = processor.process(category_keywords=["Restaurants", "Food"])
        # all_restaurants_df.to_csv("updated_all_data.csv", index=False)
        # print(f"Data processed for the entire dataset successfully! Shape: {all_restaurants_df.shape}")
        
    except FileNotFoundError:
        print(f"Dataset file '{business_file}' not found. Verify your path.")
