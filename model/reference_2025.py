"""The official 2025 tournament reference strategy, verbatim logic (200-day MA z-score boost,
excess redistributed over the last half of the window). Kept for benchmarking only."""
import numpy as np, pandas as pd
MIN_WEIGHT = 1e-5
def construct_features(df):
    df = df.copy(); df = df[['PriceUSD_coinmetrics']]
    past_price = df['PriceUSD_coinmetrics'].shift(1)
    df['ma200'] = past_price.rolling(window=200, min_periods=1).mean()
    df['std200'] = past_price.rolling(window=200, min_periods=1).std()
    return df
def compute_weights(df_window):
    features = construct_features(df_window); dates = features.index; total_days = len(features)
    weights = pd.Series(index=dates, dtype=float)
    rebalance_window = max(total_days // 2, 1); boost_alpha = 1.25
    base_weight = 1.0 / total_days; temp_weights = np.full(total_days, base_weight)
    price_array = features["PriceUSD_coinmetrics"].values; ma200_array = features["ma200"].values; std200_array = features["std200"].values
    for day_idx in range(total_days):
        price = price_array[day_idx]; ma200 = ma200_array[day_idx]; std200 = std200_array[day_idx]
        if pd.isna(ma200) or pd.isna(std200) or std200 == 0 or price >= ma200: continue
        z_score = (ma200 - price) / std200
        boosted_weight = temp_weights[day_idx] * (1 + boost_alpha * z_score); excess = boosted_weight - temp_weights[day_idx]
        start_redistribution = max(total_days - rebalance_window, day_idx + 1)
        redistribution_indices = np.arange(start_redistribution, total_days)
        if redistribution_indices.size == 0: continue
        per_day_reduction = excess / redistribution_indices.size
        if np.all(temp_weights[redistribution_indices] - per_day_reduction >= MIN_WEIGHT):
            temp_weights[day_idx] = boosted_weight; temp_weights[redistribution_indices] -= per_day_reduction
    weights.loc[dates] = temp_weights; return weights
