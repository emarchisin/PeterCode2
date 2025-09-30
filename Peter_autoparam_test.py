#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 20:06:23 2025

@author: emmamarchisin
"""

#Autoparam Testing##
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
os.chdir("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake/PeterCode2")

#load observed dara
df_obs = pd.read_csv('Peter Inputs/Obs4check/observed_data3.csv')
df_obs['datetime'] = pd.to_datetime(df_obs['datetime'], format='mixed')
df_obs = df_obs.dropna(subset=["datetime", "depth", "variable", "observation"])
df_obs = df_obs.dropna(subset=['observation']).sort_values("datetime")
df_obs['variable'] = df_obs['variable'].replace({
    "do_mgl": "do"})

def load_model_file(filepath, variable):
    df = pd.read_csv(filepath, index_col=0)   # depth as index, times as columns
    df_long = df.reset_index().melt(id_vars='index',
                                    var_name='datetime',
                                    value_name='model')
    df_long = df_long.rename(columns={'index': 'depth'})
    df_long['datetime'] = pd.to_datetime(df_long['datetime'])
    df_long['variable'] = variable
    df_long['depth'] = df_long['depth'] - 0.25 #Model produces depths at 0.25 deeper, corrects for matching
    return df_long

all_results = []

for run_id in range(1, 100):  # runs 1–99
    run_folder = f"Peter Parameterization/outputs/Run_{run_id}"
    
    if not os.path.isdir(run_folder):
        print(f"Skipping {run_folder} (not found)")
        continue
    
    try:
        # Load model  outputs 
        model_temp = load_model_file(f"{run_folder}/templab.csv", "wtemp")
        model_do   = load_model_file(f"{run_folder}/dolab.csv",  "do")
        model_doc  = load_model_file(f"{run_folder}/doclab.csv", "doc")
        df_model = pd.concat([model_temp, model_do, model_doc], ignore_index=True)
        df_model = df_model.sort_values("datetime")
        
        # Merge with obs by datetime, depth, variable
        merged = pd.merge_asof(
            df_obs,
            df_model,
            on="datetime",
            by=["variable","depth"],
            direction="nearest",
            tolerance=pd.Timedelta("1h")
        ).dropna(subset=["model"])
        
        if merged.empty:
            print(f"Run {run_id} produced no matches with obs")
            continue
        
        var_results = (
            merged.groupby("variable") #STATS PER VARIABLE
                  .apply(lambda g: pd.Series({
                      "n": len(g),
                      "r2": r2_score(g["observation"], g["model"]),
                      "rmse": np.sqrt(mean_squared_error(g["observation"], g["model"]))
                  }))
                  .reset_index()
        )
        var_results["run"] = run_id
        
        #AVERAGE ACROSS THE ALL VARIABLES
        overall_r2 = var_results["r2"].mean()
        overall_rmse = var_results["rmse"].mean()
        var_results["overall_r2"] = overall_r2
        var_results["overall_rmse"] = overall_rmse
        
        all_results.append(var_results)
    
    except Exception as e:
        print(f"Error in Run {run_id}: {e}")
        continue
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv("Peter Parameterization/run_metrics_detailed.csv", index=False)


summary = (
    results_df.groupby("run")[["overall_r2","overall_rmse"]]
              .mean()
              .reset_index()
)

best_run = summary.sort_values(["overall_r2","overall_rmse"], 
                               ascending=[False, True]).head(1)

print("Best run (overall fit):")
print(best_run)

print("\nDetailed results (first few rows):")
print(results_df.head())