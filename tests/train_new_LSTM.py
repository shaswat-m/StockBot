import os
import sys

import matplotlib.pyplot as plt

sys.path.append("../python")
from util import *

tickers = ["AAPL", "MSFT"]

# Models to compare
model_classes = [
    LSTM_Model,
    Attention_LSTM_Model,
    MultiHeadAttention_LSTM_Model,
    Transformer_Model,
    Informer_Model,
    TFT_Model,
    TCN_Model,
]

results = {}
start = "2010-01-01"
end = "2019-12-31"
for ticker in tickers:
    print(f"Running models for {ticker}")
    for model_cls in model_classes:
        print(f"Training {model_cls.__name__}")
        model = model_cls(
            tickerSymbol=ticker,
            start=start,
            end=end,
            past_history=60,
            forward_look=10,
            verbose=1,
        )
        model.full_workflow_and_plot(suffix=model_cls.__name__)
        model.plot_bot_decision(suffix=model_cls.__name__)
        plt.close()
        # Store metrics
        results[(ticker, model_cls.__name__)] = {
            "RMS_error": model.RMS_error,
            "MAE_error": model.MAE_error,
        }

for ticker in tickers:
    plt.figure()
    names = [cls.__name__ for cls in model_classes]
    rms = [results[(ticker, name)]["RMS_error"] for name in names]
    plt.bar(names, rms)
    plt.title(f"RMS Error Comparison for {ticker}")
    plt.ylabel("RMS Error")
    plt.show()
