import pandas as pd
import numpy as np
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import random


# Set random seed
np.random.seed(42)
random.seed(42)

# Data Extraction
start_date = "2019-01-01"
end_date = "2026-07-03"
symbol = "SPY"
data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1) 
data.head(3)

# Create a copy
df = data.copy()
df["Log"] = np.log(df["Close"])
df["Returns"] = (df["Log"]).pct_change()
df["Range"] = (df["High"] / df["Low"]) - 1
df.dropna(inplace=True)
print(df.head(3))

# Scale the data to fix covariance issues
scaler = StandardScaler()
X_train = scaler.fit_transform(df[["Returns", "Range"]])

# Structure Data
# X_train = df[["Returns", "Range"]]

# Fit Model
try:
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100).fit(X_train)
    print("Model Score:", hmm_model.score(X_train))
except Exception as e:
    print("Error:", e)
    
    
# Predict Hidden States
hidden_states = hmm_model.predict(X_train)

print("Hidden States (first 5):", hidden_states[:5])

# Structure prices for chart plotting
i = 0
labels_0 = []
labels_1 = []
labels_2 = []
labels_3 = []

prices = df["Close"].values.astype(float)  # Flatten the prices array
print("Correct Number of rows: ", len(prices) == len(hidden_states))

for s in hidden_states:
    if s == 0:
        labels_0.append(prices[i])
        labels_1.append(np.nan)
        labels_2.append(np.nan)
        labels_3.append(np.nan)
    if s == 1:
        labels_0.append(np.nan)
        labels_1.append(prices[i])
        labels_2.append(np.nan)
        labels_3.append(np.nan)
    if s == 2:
        labels_0.append(np.nan)
        labels_1.append(np.nan)
        labels_2.append(prices[i])
        labels_3.append(np.nan)
    if s == 3:
        labels_0.append(np.nan)
        labels_1.append(np.nan)
        labels_2.append(np.nan)
        labels_3.append(prices[i])
    i += 1
    
fig = plt.figure(figsize = (18,10))
plt.plot(labels_0, color="green")
plt.plot(labels_1, color="orange")
plt.plot(labels_2, color="red")
# plt.plot(labels_3, color="black")
plt.show()
print(len(labels_0))
