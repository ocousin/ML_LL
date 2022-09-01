# %%
import onnx 
import torch 
import os
from IPython.display import display
from preProcessing import *

# %% [markdown]
# ## Analysing the output results of FINN compiler

# %% [markdown]
# ### Directory structure

# %%
estimates_output_dir = "output_estimates_only"
! ls -l {estimates_output_dir}
! ls -l {estimates_output_dir}/report
! cat {estimates_output_dir}/report/estimate_network_performance.json

# %% [markdown]
# ### Reading estimate_network_performance

# %%
print("Reading estimate_network_performance.json ")
estimate_network_performance = pd.read_json(estimates_output_dir + '/report/estimate_network_performance.json')
display(estimate_network_performance)

# %% [markdown]
# ### Reading estimate_layer_cycles

# %%
print("Reading estimate_layer_cycles.json ")
estimate_layer_cycles = pd.read_json(estimates_output_dir + '/report/estimate_layer_cycles.json')
display(estimate_layer_cycles)

# %% [markdown]
# ### Reading estimate_layer_resources

# %%
print("Reading estimate_layer_resources.json ")
estimate_layer_cycles = pd.read_json(estimates_output_dir + '/report/estimate_layer_resources.json')
display(estimate_layer_cycles)


