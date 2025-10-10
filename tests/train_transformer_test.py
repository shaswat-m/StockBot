import os
import sys

sys.path.append("../python")
from util import *

model_single = Transformer_Model_MS(
    tickerSymbol="AAPL",
    start="2018-01-01",
    end="2023-01-01",
    past_history=60,
    forward_look=1,
    epochs=50,
    batch_size=32,
    depth=2,
    d_model=64,
    num_heads=4,
    ff_dim=128,
)

model_single.model_workflow()
pred = model_single.infer_values()
model_single.plot_predictions(save_path="../images/AAPL_Transformer_pred.png")
print("Single-ticker RMS:", model_single.RMS_error_update)

# Multi-ticker example
# model_multi = Transformer_Model_MS(
#     tickerSymbol="AAPL",
#     tickerSymbolList=["AAPL", "MSFT", "GOOGL"],
#     start="2020-01-01",
#     end="2023-01-01",
#     past_history=60,
#     forward_look=1,
#     epochs=20,
#     batch_size=32,
#     depth=2,
#     d_model=64,
#     num_heads=4,
#     ff_dim=128,
# )

# model_multi.model_workflow()
# pred_multi = model_multi.infer_values()
# model_multi.plot_predictions(save_path="../images/MultiTicker_Transformer_pred.png")
# print("Multi-ticker RMS:", model_multi.RMS_error_update)
