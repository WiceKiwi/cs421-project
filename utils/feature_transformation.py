import numpy as np
import pandas as pd
from scipy.stats import entropy

EPS = 1e-6

def aggregate_features(df):
    # --- USER-LEVEL AGGREGATION ---
    user_features = df.groupby("user").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        like_count=("rating", lambda x: (x == 10).sum()),
        dislike_count=("rating", lambda x: (x == -10).sum()),
        unknown_count=("rating", lambda x: (x == 1).sum()),
        neutral_count=("rating", lambda x: (x == 0).sum())
    ).reset_index()
    
    # Proportions and rating entropy
    user_features["like_pct"] = user_features["like_count"] / (user_features["review_count"] + EPS)
    user_features["dislike_pct"] = user_features["dislike_count"] / (user_features["review_count"] + EPS)
    user_features["unknown_pct"] = user_features["unknown_count"] / (user_features["review_count"] + EPS)
    user_features["neutral_pct"] = user_features["neutral_count"] / (user_features["review_count"] + EPS)
    
    def calc_entropy(row):
        probs = [row["like_pct"], row["dislike_pct"], row["unknown_pct"], row["neutral_pct"]]
        probs = [p for p in probs if p > 0]
        return entropy(probs) if probs else 0

    user_features["rating_entropy"] = user_features.apply(calc_entropy, axis=1)
    
    # --- MOVIE POPULARITY FEATURES ---
    movie_popularity = df.groupby("item").size().reset_index(name="movie_popularity")
    df_with_pop = df.merge(movie_popularity, on="item")
    pop_features = df_with_pop.groupby("user").agg(
        avg_movie_popularity=("movie_popularity", "mean"),
        std_movie_popularity=("movie_popularity", "std"),
        min_movie_popularity=("movie_popularity", "min"),
        max_movie_popularity=("movie_popularity", "max")
    ).reset_index()
    
    # Rare movie stats
    movie_pop = df['item'].value_counts().reset_index()
    movie_pop.columns = ['item', 'popularity']
    threshold = movie_pop['popularity'].quantile(0.1)
    rare_movies = movie_pop[movie_pop['popularity'] <= threshold]['item'].values
    df['is_rare_movie'] = df['item'].isin(rare_movies).astype(int)
    rare_stats = df.groupby('user')['is_rare_movie'].mean().reset_index(name='rare_movies_watched_pct')
    user_features = user_features.merge(rare_stats, on='user', how='left')
    
    # --- UNIQUE MOVIE COUNT ---
    unique_items = df.groupby("user")["item"].nunique().reset_index().rename(columns={"item": "unique_movies"})
    
    # --- DEVIATION FROM POPULATION FEATURES ---
    movie_avg_rating = df.groupby("item")["rating"].mean().reset_index(name="movie_avg_rating")
    df_with_avg = df.merge(movie_avg_rating, on="item")
    df_with_avg["rating_deviation"] = df_with_avg["rating"] - df_with_avg["movie_avg_rating"]
    df_with_avg["abs_rating_deviation"] = np.abs(df_with_avg["rating_deviation"])
    deviation_features = df_with_avg.groupby("user").agg(
        mean_deviation=("rating_deviation", "mean"),
        std_deviation=("rating_deviation", "std"),
        mean_abs_deviation=("abs_rating_deviation", "mean"),
        max_abs_deviation=("abs_rating_deviation", "max")
    ).reset_index()
    
    # --- SEQUENTIAL PATTERN FEATURES ---
    df_sorted = df.sort_values(["user", "item"]).copy()
    df_sorted["next_rating"] = df_sorted.groupby("user")["rating"].shift(-1)
    df_sorted["rating_diff"] = df_sorted["next_rating"] - df_sorted["rating"]
    df_sorted["abs_rating_diff"] = np.abs(df_sorted["rating_diff"])
    df_sorted = df_sorted.dropna(subset=["rating_diff"])
    
    df_sorted["rating_direction"] = df_sorted["rating_diff"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df_sorted['direction_switch'] = df_sorted["rating_direction"] != df_sorted.groupby("user")["rating_direction"].shift(1)
    switch_count = df_sorted.groupby("user")["direction_switch"].sum().reset_index(name="change_direction_count")
    user_features = user_features.merge(switch_count, on="user", how="left")
    
    sequence_features = df_sorted.groupby("user").agg(
        mean_rating_diff=("rating_diff", "mean"),
        std_rating_diff=("rating_diff", "std"),
        mean_abs_rating_diff=("abs_rating_diff", "mean"),
        max_abs_rating_diff=("abs_rating_diff", "max"),
        rating_changes_count=("rating_diff", lambda x: (x != 0).sum())
    ).reset_index()
    sequence_features["rating_changes_pct"] = sequence_features["rating_changes_count"] / (
        user_features.set_index("user")["review_count"] - 1 + EPS
    ).reindex(sequence_features["user"]).values
    
    # --- COMBINING FEATURES ---
    all_features = user_features.merge(pop_features, on="user", how="left")
    all_features = all_features.merge(unique_items, on="user", how="left")
    all_features = all_features.merge(deviation_features, on="user", how="left")
    all_features = all_features.merge(sequence_features, on="user", how="left")
    all_features = all_features.fillna(0)
    
    # --- ADVANCED INTERACTION FEATURES ---
    all_features["like_dislike_ratio"] = all_features["like_count"] / (all_features["dislike_count"] + EPS)
    all_features["rating_range"] = all_features["max_abs_rating_diff"]
    all_features["popularity_vs_deviation"] = all_features["avg_movie_popularity"] * all_features["mean_abs_deviation"]
    all_features["entropy_by_count"] = all_features["rating_entropy"] * np.log1p(all_features["review_count"])
    
    # --- REVIEW COUNT BINNING ---
    all_features["review_count_bin"] = pd.qcut(all_features["review_count"], q=5, labels=False, duplicates="drop")
    
    # --- ADDITIONAL MOVIE ITEM FEATURES ---
    min_max_df = df.groupby("user")["item"].agg(
        min_movie="min", max_movie="max", median_movie="median", variance_movie="var"
    ).reset_index()
    all_features = all_features.merge(min_max_df, on="user", how="left")
    
    df["item_rating"] = df["item"] * df["rating"]
    sum_rating = df.groupby("user")["rating"].sum().reset_index(name="sum_rating")
    sum_product = df.groupby("user")["item_rating"].sum().reset_index(name="sum_item_rating")
    all_features = all_features.merge(sum_product, on="user", how="left")
    all_features = all_features.merge(sum_rating, on="user", how="left")
    
    all_features["average_product"] = all_features["sum_item_rating"] / all_features["review_count"]
    all_features["product_above_zero"] = (all_features["sum_item_rating"] > 0).astype(int)
    all_features["sum_above_zero"] = (all_features["sum_rating"] > 0).astype(int)
    all_features["avg_product_vs_avg_rating"] = all_features["average_product"] / (all_features["avg_rating"] + EPS)
    
    for col in [c for c in all_features.columns if c.endswith("_ratio")]:
        all_features[col] = all_features[col].round(2)
    
    df_sorted = df.sort_values(by=["user", "item"]).copy()
    df_sorted["item_diff"] = df_sorted.groupby("user")["item"].diff().fillna(0)
    gap_stats = df_sorted.groupby("user")["item_diff"].agg(["mean", "std", "max"]).reset_index()
    gap_stats.columns = ["user", "gap_mean", "gap_std", "gap_max"]
    all_features = all_features.merge(gap_stats, on="user", how="left")
    
    return all_features