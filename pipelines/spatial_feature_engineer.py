import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split

class SpatialFeatureEngineer:
    def __init__(self, df):
        """
        Initialize the engineer with the DataFrame output from YelpDataProcessor.
        """
        self.df = df

    def compute_local_features(self, target_coords, target_cats, ref_tree, ref_cats, ref_stars, ref_surv, ref_revs, exclude_self=False):
        MAX_RADIUS_KM = 3.0
        max_rad = MAX_RADIUS_KM / 6371.0
        M = len(target_coords)
        
        out = {}
        for r in [0.5, 1.0, 3.0]:
            r_str = f"{r}km"
            out[f'count_all_{r_str}'] = np.zeros(M)
            out[f'avg_rating_all_{r_str}'] = np.zeros(M)
            out[f'survival_all_{r_str}'] = np.zeros(M)
            out[f'diversity_{r_str}'] = np.zeros(M)
            
            out[f'count_same_cat_{r_str}'] = np.zeros(M)
            out[f'avg_rating_same_cat_{r_str}'] = np.zeros(M)
            out[f'survival_same_cat_{r_str}'] = np.zeros(M)

        out['dist_nearest_same_cat'] = np.zeros(M)
        
        global_stars = ref_stars.mean()
        global_surv = ref_surv.mean() 
        
        idxs_list, dists_list = ref_tree.query_radius(target_coords, r=max_rad, return_distance=True)
        
        for i in range(M):
            idxs_3km = idxs_list[i]
            dists_3km = dists_list[i] * 6371.0
            
            if exclude_self:
                mask = (idxs_3km != i)
                idxs_3km = idxs_3km[mask]
                dists_3km = dists_3km[mask]
                
            neighbor_cats_mx = ref_cats[idxs_3km]
            point_cat = target_cats[i]
            shared_mask_3km = (neighbor_cats_mx @ point_cat) > 0
            
            same_cat_idxs_3km = idxs_3km[shared_mask_3km]
            same_cat_dists_3km = dists_3km[shared_mask_3km]
            
            if len(same_cat_dists_3km) > 0:
                out['dist_nearest_same_cat'][i] = same_cat_dists_3km.min()
            else:
                out['dist_nearest_same_cat'][i] = MAX_RADIUS_KM
                
            for r in [0.5, 1.0, 3.0]:
                r_str = f"{r}km"
                mask_r = dists_3km <= r
                r_idxs = idxs_3km[mask_r]
                n_all = len(r_idxs)
                
                out[f'count_all_{r_str}'][i] = n_all
                if n_all > 0:
                    out[f'avg_rating_all_{r_str}'][i] = ref_stars[r_idxs].mean()
                    out[f'survival_all_{r_str}'][i] = ref_surv[r_idxs].mean()
                    out[f'diversity_{r_str}'][i] = (ref_cats[r_idxs].sum(axis=0) > 0).sum()
                else:
                    out[f'avg_rating_all_{r_str}'][i] = global_stars
                    out[f'survival_all_{r_str}'][i] = global_surv
                    out[f'diversity_{r_str}'][i] = 0
                
                same_cat_mask_r = same_cat_dists_3km <= r
                r_same_idxs = same_cat_idxs_3km[same_cat_mask_r]
                n_same = len(r_same_idxs)
                
                out[f'count_same_cat_{r_str}'][i] = n_same
                if n_same > 0:
                    out[f'avg_rating_same_cat_{r_str}'][i] = ref_stars[r_same_idxs].mean()
                    out[f'survival_same_cat_{r_str}'][i] = ref_surv[r_same_idxs].mean()
                else:
                    out[f'avg_rating_same_cat_{r_str}'][i] = global_stars
                    out[f'survival_same_cat_{r_str}'][i] = global_surv

        # Feature Transformations
        for r in [0.5, 1.0, 3.0]:
            r_str = f"{r}km"
            
            # Stabilize counts
            out[f'log_count_all_{r_str}'] = np.log1p(np.clip(out[f'count_all_{r_str}'], 0, 5000))
            out[f'log_count_same_cat_{r_str}'] = np.log1p(out[f'count_same_cat_{r_str}'])
            
            # Binary checks
            out[f'has_any_{r_str}'] = (out[f'count_all_{r_str}'] > 0).astype(float)
            out[f'has_same_cat_{r_str}'] = (out[f'count_same_cat_{r_str}'] > 0).astype(float)
            out[f'low_count_{r_str}'] = (out[f'count_all_{r_str}'] < 5).astype(float)
            
            # Ratios
            out[f'same_cat_ratio_{r_str}'] = out[f'count_same_cat_{r_str}'] / (out[f'count_all_{r_str}'] + 1.0)
            
            # Gaps vs Global Priors
            out[f'rating_gap_global_{r_str}'] = out[f'avg_rating_all_{r_str}'] - global_stars
            out[f'survival_gap_global_{r_str}'] = out[f'survival_all_{r_str}'] - global_surv
            out[f'same_cat_rating_gap_global_{r_str}'] = out[f'avg_rating_same_cat_{r_str}'] - global_stars
            out[f'same_cat_survival_gap_global_{r_str}'] = out[f'survival_same_cat_{r_str}'] - global_surv
            
            # Gaps vs Local Context
            out[f'same_cat_vs_local_rating_gap_{r_str}'] = out[f'avg_rating_same_cat_{r_str}'] - out[f'avg_rating_all_{r_str}']
            out[f'same_cat_vs_local_survival_gap_{r_str}'] = out[f'survival_same_cat_{r_str}'] - out[f'survival_all_{r_str}']
            
            # Diversity
            out[f'log_diversity_{r_str}'] = np.log1p(out[f'diversity_{r_str}'])
            
        out['log_dist_nearest_same_cat'] = np.log1p(out['dist_nearest_same_cat'])
        
        return pd.DataFrame(out)

    def split_and_engineer_spatial_features(self):
        print("Splitting data and engineering spatial features...")
        cat_cols = [c for c in self.df.columns if c.startswith('cat_')]
        
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        train_coords = np.radians(train_df[['latitude', 'longitude']].values)
        train_tree = BallTree(train_coords, metric='haversine')
        
        ref_cats = train_df[cat_cols].values
        ref_stars = train_df['stars'].values
        # Using is_open directly for survival calculations
        ref_surv = train_df['is_open'].values
        ref_revs = train_df['review_count'].values
        
        print("Computing training spatial features (excluding self)...")
        train_feats = self.compute_local_features(train_coords, ref_cats, train_tree, ref_cats, ref_stars, ref_surv, ref_revs, exclude_self=True)
        
        test_coords = np.radians(test_df[['latitude', 'longitude']].values)
        test_cats = test_df[cat_cols].values
        
        print("Computing testing spatial features (using train as reference)...")
        test_feats = self.compute_local_features(test_coords, test_cats, train_tree, ref_cats, ref_stars, ref_surv, ref_revs, exclude_self=False)
        
        train_spatial_df = pd.concat([train_df, train_feats], axis=1).fillna(0)
        test_spatial_df = pd.concat([test_df, test_feats], axis=1).fillna(0)
        
        return train_spatial_df, test_spatial_df

    def engineer_single_target(self, target_coord, target_cat, reference_df):
        """
        [PLACEHOLDER] 
        Calculates spatial features on-the-fly for real-time live map requests.
        Takes a single (lat, lon) target pair, and uses a pre-loaded reference spatial tree
        to instantly compute the gaps and features without rebuilding the index.
        """
        pass
