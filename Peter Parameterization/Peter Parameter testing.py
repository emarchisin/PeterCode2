#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 17:46:19 2025

@author: emmamarchisin
"""
#Parameterization output analysis 

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import os
os.chdir("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake")
#sys.path.append("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake")

#obs data in long format, run data in array with no labels depth across columns and rows as times
obs = pd.read_csv("Peter Inputs/observed_data2", parse_dates=["datetime"]) 
obs.columns = obs.columns.str.lower()
obs['datetime'] = pd.to_datetime(obs['datetime'], errors='coerce')
output_base = "Peter Parameterization/outputs"

nx = 19*2
zmax = 19
depth_grid = np.linspace(0, zmax, nx)
#model_depths = np.round(depth_grid, 2)  # for matching
model_depths = np.arange(0, 19.5, 0.5)

#create time grid based on model assumption
model_start = pd.Timestamp("2024-06-10 09:00:00")
n_time_steps = len(pd.read_csv(os.path.join(output_base, "Run_1/temp.csv"), header=None))
time_grid = pd.date_range(start=model_start, periods=n_time_steps, freq='H')

r2_results = []

#run through files
for run_folder in sorted(os.listdir(output_base)):
    run_path = os.path.join(output_base, run_folder)
    if not os.path.isdir(run_path):
        continue

    run_number = int(run_folder.split("_")[-1])
    print(f"Processing Run {run_number}")

    try:
        # Load model output
        model_data = {
            "wtemp": pd.read_csv(os.path.join(run_path, "temp.csv"), header=None).values,
            "do": pd.read_csv(os.path.join(run_path, "do_mgL.csv"), header=None).values,
            "doc": pd.read_csv(os.path.join(run_path, "doc_mgL.csv"), header=None).values,
            "secchi": pd.read_csv(os.path.join(run_path, "secchi.csv"), header=None).values #"do": pd.read_csv(os.path.join(run_path, "do_mgL.csv"), header=None).values.T,
        }
         ##checking 
        for var in ["wtemp", "do", "doc", "secchi"]:
            obs_var = obs.dropna(subset=[var, "depth", "datetime"])
            n_total = len(obs_var)
            matched = 0

            for _, row in obs_var.iterrows():
                obs_time = row['datetime']
                obs_depth = row['depth']

                if obs_depth not in model_depths:
                    continue
                if obs_time < time_grid[0] or obs_time > time_grid[-1]:
                    continue

            matched += 1

        print(f"{var}: {matched} of {n_total} observations matched model grid")
        for _, row in obs_var.iterrows():
            obs_time = row['datetime']
            obs_depth = row['depth']

            if obs_depth not in model_depths:
                print(f"Depth mismatch: {obs_depth}")
                continue
            if obs_time < time_grid[0] or obs_time > time_grid[-1]:
                print(f"Time mismatch: {obs_time}")
                continue

            matched += 1
            #end of checking

        r2_scores = {"Run": run_number}

        # Compare each variable
        for var in ["wtemp", "do", "doc", "secchi"]:
            obs_var = obs.dropna(subset=[var, "depth", "datetime"])  # only drop missing for this var
            preds = []
            actuals = []

            for _, row in obs_var.iterrows():
                obs_time = row['datetime']
                obs_depth = row['depth']

                # Match exact depth
                if obs_depth not in model_depths:
                    continue  # skip if not in model output

                # Skip if out of time range
                if obs_time < time_grid[0] or obs_time > time_grid[-1]:
                    continue

                # Find surrounding times
                time_before = time_grid[time_grid <= obs_time].max()
                time_after = time_grid[time_grid >= obs_time].min()

                i_before = time_grid.get_loc(time_before)
                i_after = time_grid.get_loc(time_after)

                d_index = np.where(model_depths == obs_depth)[0][0]

                val_before = model_data[var][i_before, d_index]
                val_after = model_data[var][i_after, d_index]

                # Linear interpolation in time
                if time_before == time_after:
                    val_interp = val_before
                else:
                    dt_total = (time_after - time_before).total_seconds()
                    dt_frac = (obs_time - time_before).total_seconds() / dt_total
                    val_interp = val_before + dt_frac * (val_after - val_before)

                preds.append(val_interp)
                actuals.append(row[var])

            # Calculate R²
            if len(preds) > 0:
                r2_scores[f"R2_{var}"] = r2_score(actuals, preds)
            else:
                r2_scores[f"R2_{var}"] = np.nan

        # Average R² across variables
        valid_r2s = [v for k, v in r2_scores.items() if k.startswith("R2_") and not np.isnan(v)]
        r2_scores["Average_R2"] = np.mean(valid_r2s) if valid_r2s else np.nan

        r2_results.append(r2_scores)

    except Exception as e:
        print(f"Run {run_number} failed: {e}")
        continue
