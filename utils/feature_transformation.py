import numpy as np
import pandas as pd
from scipy.stats import entropy, skew

EPS = 1e-6

def aggregate_features(df):
    # ================= USER-LEVEL AGGREGATION =================
    user_features = df.groupby("user").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        like_count=("rating", lambda x: (x == 10).sum()),
        dislike_count=("rating", lambda x: (x == -10).sum()),
        unknown_count=("rating", lambda x: (x == 1).sum()),
        neutral_count=("rating", lambda x: (x == 0).sum())
    ).reset_index()

    # Proportions and entropy
    user_features["like_pct"] = user_features["like_count"] / (user_features["review_count"] + EPS)
    user_features["dislike_pct"] = user_features["dislike_count"] / (user_features["review_count"] + EPS)
    user_features["unknown_pct"] = user_features["unknown_count"] / (user_features["review_count"] + EPS)
    user_features["neutral_pct"] = user_features["neutral_count"] / (user_features["review_count"] + EPS)

    def calc_entropy(row):
        probs = [row["like_pct"], row["dislike_pct"], row["unknown_pct"], row["neutral_pct"]]
        probs = [p for p in probs if p > 0]
        return entropy(probs) if probs else 0

    user_features["rating_entropy"] = user_features.apply(calc_entropy, axis=1)

    # ================= MOVIE POPULARITY FEATURES =================
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
    threshold = movie_pop['popularity'].quantile(0.05)
    rare_movies = movie_pop[movie_pop['popularity'] <= threshold]['item'].values
    df['is_rare_movie'] = df['item'].isin(rare_movies).astype(int)
    rare_stats = df.groupby('user')['is_rare_movie'].mean().reset_index(name='rare_movies_watched_pct')
    user_features = user_features.merge(rare_stats, on='user', how='left')

    # ================= UNIQUE MOVIE COUNT =================
    unique_items = df.groupby("user")["item"].nunique().reset_index().rename(columns={"item": "unique_movies"})

    # ================= DEVIATION FROM POPULATION FEATURES =================
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

    # ================= SEQUENTIAL PATTERN FEATURES =================
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

    # ================= Z-RATING STATS =================
    movie_stats = df.groupby("item")["rating"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "m_mean", "std": "m_std"})
    df = df.merge(movie_stats, on="item", how="left")
    df["z_rating"] = (df["rating"] - df["m_mean"]) / (df["m_std"] + EPS)

    z_stats = df.groupby("user")["z_rating"].agg(["mean", "std", "max", "min", "median", "skew"]).reset_index()
    z_stats.columns = ["user"] + [f"z_rating_{stat}" for stat in z_stats.columns if stat != "user"]

    # ================= POPULARITY THRESHOLD BIAS =================
    df = df.merge(movie_popularity, on="item", how="left")
    pop_threshold = movie_popularity["movie_popularity"].quantile(0.75)
    rare_threshold = movie_popularity["movie_popularity"].quantile(0.25)

    df["likes_popular"] = ((df["movie_popularity"] > pop_threshold) & (df["rating"] == 10)).astype(int)
    df["likes_rare"] = ((df["movie_popularity"] < rare_threshold) & (df["rating"] == 10)).astype(int)
    df["dislikes_popular"] = ((df["movie_popularity"] > pop_threshold) & (df["rating"] == -10)).astype(int)
    df["dislikes_rare"] = ((df["movie_popularity"] < rare_threshold) & (df["rating"] == -10)).astype(int)
    df["neutral_popular"] = ((df["movie_popularity"] > pop_threshold) & (df["rating"] == 0)).astype(int)
    df["neutral_rare"] = ((df["movie_popularity"] < rare_threshold) & (df["rating"] == 0)).astype(int)
    df["unknown_popular"] = ((df["movie_popularity"] > pop_threshold) & (df["rating"] == 1)).astype(int)
    df["unknown_rare"] = ((df["movie_popularity"] < rare_threshold) & (df["rating"] == 1)).astype(int)

    pop_bias = df.groupby("user")[[
        "likes_popular", "likes_rare",
        "dislikes_popular", "dislikes_rare",
        "neutral_popular", "neutral_rare",
        "unknown_popular", "unknown_rare"
    ]].mean().reset_index()

    # ================= INTERACTION ENTROPY =================
    def interaction_entropy(x):
        probs = x.value_counts(normalize=True)
        return entropy(probs)

    interaction_ent = df.groupby("user")[["item", "rating"]].apply(
        lambda g: interaction_entropy(g["item"].astype(str) + "_" + g["rating"].astype(str))
    ).reset_index(name="interaction_entropy")

    # ================= MERGE ALL FEATURES =================
    all_features = user_features.merge(pop_features, on="user", how="left")
    all_features = all_features.merge(unique_items, on="user", how="left")
    all_features = all_features.merge(deviation_features, on="user", how="left")
    all_features = all_features.merge(sequence_features, on="user", how="left")
    all_features = all_features.merge(z_stats, on="user", how="left")
    all_features = all_features.merge(pop_bias, on="user", how="left")
    all_features = all_features.merge(interaction_ent, on="user", how="left")

    # ================= ADVANCED INTERACTIONS =================
    all_features["like_dislike_ratio"] = all_features["like_count"] / (all_features["dislike_count"] + EPS)
    all_features["rating_range"] = all_features["max_abs_rating_diff"]
    all_features["popularity_vs_deviation"] = all_features["avg_movie_popularity"] * all_features["mean_abs_deviation"]
    all_features["entropy_by_count"] = all_features["rating_entropy"] * np.log1p(all_features["review_count"])

    # ================= MOVIE ITEM POSITION STATS =================
    all_features["review_count_bin"] = pd.qcut(all_features["review_count"], q=5, labels=False, duplicates="drop")
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

    # ================= GAP STATS =================
    df_sorted = df.sort_values(by=["user", "item"]).copy()
    df_sorted["item_diff"] = df_sorted.groupby("user")["item"].diff().fillna(0)
    gap_stats = df_sorted.groupby("user")["item_diff"].agg(["mean", "std", "max"]).reset_index()
    gap_stats.columns = ["user", "gap_mean", "gap_std", "gap_max"]
    all_features = all_features.merge(gap_stats, on="user", how="left")

    # ================= POPULARITY DISTRIBUTION =================
    movie_popularity = df["item"].value_counts(normalize=True)
    movie_percentile = movie_popularity.rank(pct=True)
    df["movie_pop_percentile"] = df["item"].map(movie_percentile)
    pop_diff = df.groupby("user")["movie_pop_percentile"].agg(["mean", "std"]).reset_index()
    pop_diff.columns = ["user", "user_pop_percentile_mean", "user_pop_percentile_std"]
    all_features = all_features.merge(pop_diff, on="user", how="left")

    rating_distr = df.pivot_table(index="user", columns="rating", aggfunc="size", fill_value=0)
    rating_distr = rating_distr.div(rating_distr.sum(axis=1), axis=0).reset_index()
    rating_distr.columns = ["user"] + [f"rating_pct_{int(col)}" for col in rating_distr.columns if col != "user"]
    all_features = all_features.merge(rating_distr, on="user", how="left")

    # ================= MOVIE-RELATED AGGREGATES =================
    movie_stats = df.groupby("item").agg(
        item_rating_mean=("rating", "mean"),
        item_rating_std=("rating", "std"),
        item_rating_median=("rating", "median"),
        item_rating_skew=("rating", lambda x: skew(x.dropna()) if x.dropna().std() > 1e-6 else 0),
        item_review_count=("rating", "count")
    ).reset_index()
    df = df.merge(movie_stats, on="item", how="left")

    user_movie_agg = df.groupby("user").agg(
        mean_item_rating_mean=("item_rating_mean", "mean"),
        std_item_rating_mean=("item_rating_mean", "std"),
        mean_item_rating_std=("item_rating_std", "mean"),
        mean_item_rating_median=("item_rating_median", "mean"),
        mean_item_rating_skew=("item_rating_skew", "mean"),
        mean_item_review_count=("item_review_count", "mean"),
        max_item_review_count=("item_review_count", "max"),
        min_item_review_count=("item_review_count", "min")
    ).reset_index()

    all_features = all_features.merge(user_movie_agg, on="user", how="left")

    all_features["item_mean_vs_user_avg"] = all_features["mean_item_rating_mean"] - all_features["avg_rating"]
    all_features["item_skew_bias"] = all_features["mean_item_rating_skew"]
    all_features["normalized_movie_popularity"] = all_features["avg_movie_popularity"] / (all_features["mean_item_review_count"] + EPS)
    all_features["popularity_skew"] = all_features["max_movie_popularity"] - all_features["min_movie_popularity"]
    all_features["rating_polarity"] = all_features["like_pct"] - all_features["dislike_pct"]
    all_features["activity_weighted_skew"] = all_features["z_rating_skew"] * np.log1p(all_features["review_count"])
    all_features["switch_pct"] = all_features["change_direction_count"] / (all_features["review_count"] + EPS)

    # ================= DOMINANT RATING FEATURES =================
    dominant_rating = df.groupby(["user", "rating"]).size().reset_index(name="count")
    dominant_rating = dominant_rating.sort_values(["user", "count"], ascending=[True, False])
    dominant_rating = dominant_rating.drop_duplicates("user")

    rating_mode = dominant_rating[["user", "rating"]].rename(columns={"rating": "dominant_rating"})
    rating_mode_count = dominant_rating[["user", "count"]].rename(columns={"count": "dominant_rating_count"})
    rating_total = df.groupby("user").size().reset_index(name="total_rating_count")

    rating_mode = rating_mode.merge(rating_mode_count, on="user")
    rating_mode = rating_mode.merge(rating_total, on="user")
    rating_mode["dominance_ratio"] = rating_mode["dominant_rating_count"] / (rating_mode["total_rating_count"] + EPS)

    all_features = all_features.merge(rating_mode[["user", "dominant_rating", "dominance_ratio"]], on="user", how="left")

    # ================= HANDLE MISSING VALUES =================
    safe_cols = [col for col in all_features.columns if "count" in col or "pct" in col or col.startswith("is_") or col.endswith("_flag")]
    all_features[safe_cols] = all_features[safe_cols].fillna(0)
    all_features = all_features.fillna(all_features.median(numeric_only=True))
    


    return all_features
