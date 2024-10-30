# %% [markdown]
# # SLEGO Project: UNSW CSE PhD Research - Alan Siu Lung Ng
# https://github.com/alanntl/SLEGO-Project

# %%
# First, import the environment_setup module
import environment_setup
import os
# see current filder
os.getcwd()
repo_path = os.getcwd()

# Setup environment with explicit local_repo_path
config = environment_setup.setup_environment(use_local_repo=True, local_repo_path=repo_path)


# %% [markdown]
# # SLEGO APP

# %%
#if jupyter use display , else print
if 'get_ipython' in globals():
    from IPython.display import display
    display(config)
else:
    print(config)

# %%

import panel as pn

# Kill all existing servers
pn.state.kill_all_servers()


import slegoapp
# Create and run the new app
slego_app = slegoapp.SLEGOApp(config)
slego_app.run([ "func_yfinance.py", "func_finadvisor.py"])

# %%



