"""
# Part  of localization phase
"""
import os
import sys
import configparser
from datetime import datetime

config_name = sys.argv[1]

lemon_cfg = configparser.ConfigParser()
lemon_cfg.read(f"./config/{config_name}")
parameters = lemon_cfg['parameters']
python_prefix = parameters['python_prefix'].rstrip("/")

# 1. get unique inconsistency
start_time = datetime.now()
print("Localization Starts!")
print("\n\nPhase1: Get Unique Inconsistency")
get_unique_inconsistency = f"{python_prefix}/lemon/bin/python -u -m scripts.localization.get_unique_inconsistency {config_name}"
os.system(get_unique_inconsistency)

# 2. localization
print("\n\nPhase2: Localize")
localize = f"{python_prefix}/lemon/bin/python  -u -m run.localize_lemon {config_name}"
os.system(localize)

# 3. get suspected bugs
print("\n\nPhase3: Suspected bugs analysis")
get_suspecte_bugs = f"{python_prefix}/lemon/bin/python  -u -m scripts.localization.suspected_bugs_detector {config_name}"
os.system(get_suspecte_bugs)

print("Localization finishes!")
print(f"Localization time cost: {datetime.now() - start_time}")