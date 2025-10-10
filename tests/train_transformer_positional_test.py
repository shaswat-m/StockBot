import os
import sys

sys.path.append("../python")
from util import *

vanilla_model = Transformer_Model_MS(
    tickerSymbol="AAPL",
    start="2018-01-01",
    end="2023-01-01",
    epochs=50,
    use_learnable_pos_encoding=False,
)
vanilla_model.model_workflow()
vanilla_pred = vanilla_model.infer_values()
vanilla_model.plot_predictions("../images/AAPL_Transformer_vanilla.png")
print("Vanilla RMS:", vanilla_model.RMS_error_update)

# Enhanced Transformer (learnable positional encoding + dropout scheduling)
enhanced_model = Transformer_Model_MS(
    tickerSymbol="AAPL",
    start="2018-01-01",
    end="2023-01-01",
    epochs=50,
    use_learnable_pos_encoding=True,
)
enhanced_model.model_workflow()
enhanced_pred = enhanced_model.infer_values()
enhanced_model.plot_predictions("../images/AAPL_Transformer_enhanced.png")
print("Enhanced RMS:", enhanced_model.RMS_error_update)
