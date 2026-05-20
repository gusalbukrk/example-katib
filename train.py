# train.py
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def main():
    if len(sys.argv) < 5:
        print("Usage: train.py <max_depth> <min_samples_split> <window_size> <min_samples_leaf>")
        sys.exit(1)

    max_depth = int(sys.argv[1])
    min_samples_split = int(sys.argv[2])
    window_size = int(sys.argv[3])
    min_samples_leaf = int(sys.argv[4])

    # 1. Fetch Data
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    time_series = pd.read_csv(url)['Temp'].values

    # 2. Dynamic Sliding Window
    # transforms a raw 1D time-series array into a tabular format using a sliding window
    # 3650 rows => 3640x10 matrix
    X_seq, y_seq = [], []
    for i in range(len(time_series) - window_size):
        X_seq.append(time_series[i : i + window_size])
        y_seq.append(time_series[i + window_size])

    X_arr, y_arr = np.array(X_seq), np.array(y_seq)

    # 3. Time-Series Cross-Validation Setup (5-Fold Expanding Window)
    tscv = TimeSeriesSplit(n_splits=5)
    train_mse_scores = []
    train_mae_scores = []
    test_mse_scores = []
    test_mae_scores = []

    # 4. Cross-Validation Training Loop
    for train_index, test_index in tscv.split(X_arr):
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Track metrics per split
        train_mse_scores.append(mean_squared_error(y_train, model.predict(X_train)))
        train_mae_scores.append(mean_absolute_error(y_train, model.predict(X_train)))
        test_mse_scores.append(mean_squared_error(y_test, model.predict(X_test)))
        test_mae_scores.append(mean_absolute_error(y_test, model.predict(X_test)))

    import time
    time.sleep(2)  # Small operational buffer for local cluster synchronization

    # 5. Output Aggregated Metrics for Katib parsing
    print(f"train-mse={np.mean(train_mse_scores)}")
    print(f"train-mae={np.mean(train_mae_scores)}")
    print(f"mse={np.mean(test_mse_scores)}")
    print(f"mae={np.mean(test_mae_scores)}")

if __name__ == "__main__":
    main()
