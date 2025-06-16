import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt, isnan
import math
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import random
import tempfile
import gc
import sys
sys.path.append("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake")

tempfile.tempdir = "/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake/RT_temp"
gc.enable()

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
os.chdir("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake")
from Peter24_processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens #heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 19 # maximum lake depth
nx = 19 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = "/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake/Peter Inputs/peter_bath.csv",
                            dx = dx, nx = nx)
                            
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = "/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/All Cascade Lakes/Cascade Inputs/Cascade_2024_all_variables_for_1DAEMpy.csv",
                    secchifile = None, 
                    windfactor = 1.0
                    )
                     
## time step discretization 
n_years = 0.307
hydrodynamic_timestep = 24 * dt
total_runtime =  (365 * n_years) * hydrodynamic_timestep/dt  
startTime =  1695 # DOY in 2016 * 24 hours
endTime =  4381# * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)
#pd.DataFrame(times).to_csv("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake/Peter Parameterization/Test/times.csv", index = False)
## here we define our initial profile
u_ini = initial_profile(initfile = 'Peter Inputs/peter_wtemph.csv', nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate)

wq_ini = wq_initial_profile(initfile = 'Peter Inputs/peter_driver.csv', nx = nx, dx = dx, #tp needs whole water column
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)

tp_boundary = provide_phosphorus(tpfile =  'Peter Inputs/peter_tp.csv', #avg epi good here
                                 startingDate = startingDate,
                                 startTime = startTime)

tp_boundary = tp_boundary.dropna(subset=['observation'])

#Start = datetime.datetime.now()

# Default Values for run 1
#sw_factor = 1.0 #maybe not, +-20%
#wind_factor = 1.2 #see sw
#at_factor = 1.0 #see sw
#turb_factor = 1.0 #see sw
p_max = 3#/86400 #0.5 - 5
IP = 3e-5#/86400 #0.1, 3e-5 #1e-5, 6e-5
theta_npp = 1.08 #1.08 #1.04-1.2
theta_r = 1.08 #1.08 #see theta npp
sed_sink = -0.5 #/ 86400 #0.001-0.1
#k_half = 0.5 #0.1-0.5, related to DO
resp_docr = 0.003 #/ 86400 # 0.005-0.0001
resp_docl = 0.05 #/ 86400 # 0.01-0.1
resp_poc = 0.3#/ 86400 #0.05-0.5
settling_rate = 0.7 #/ 86400 #0.1-1
sediment_rate = 0.7 #/ 86400 #0.1-1
#piston_velocity = 1.0 #/ 86400 #not used
light_water = 0.21 #need to change for Paul
light_doc = 0.02 #.01-.04 #maybe manually fit
light_poc = 0.5 #.2-1

n_runs = 99

if os.path.isfile("Peter Parameterization/results.csv"):
    print("Parameterization file found")
else:
    #p_max_col = [x+p_max for x in [p_max * random.uniform(-1, 1) for _ in range(n_runs-1)]]
    # p_max_col.insert(0, p_max)
    p_max_col = [random.uniform(3.0, 3.5) for _ in range(n_runs-1)]
    p_max_col.insert(0, p_max)
    IP_col = [random.uniform(1.57e-5, 5.2e-5) for _ in range(n_runs-1)]
    IP_col.insert(0, IP)
    theta_npp_col = [random.uniform(1.08, 1.16) for _ in range(n_runs-1)]
    theta_npp_col.insert(0, theta_npp)
    theta_r_col = [random.uniform(1.071, 1.09) for _ in range(n_runs-1)]
    theta_r_col.insert(0, theta_r)
    sed_sink_col = [random.uniform(-0.6, -0.22) for _ in range(n_runs-1)]
    sed_sink_col.insert(0, sed_sink)
    resp_docl_col = [random.uniform(0.007, 0.009) for _ in range(n_runs-1)]
    resp_docl_col.insert(0, resp_docl)
    resp_docr_col = [random.uniform(0.007,0.01) for _ in range(n_runs-1)]
    resp_docr_col.insert(0, resp_docr)
    resp_poc_col = [random.uniform(0.25, 0.41) for _ in range(n_runs-1)]
    resp_poc_col.insert(0, resp_poc)
    settling_rate_col = [random.uniform(0.75, 0.95) for _ in range(n_runs-1)]
    settling_rate_col.insert(0, settling_rate)
    sediment_rate_col = [random.uniform(0.65, 0.76) for _ in range(n_runs-1)]
    sediment_rate_col.insert(0, sediment_rate)
    light_poc_col = [random.uniform(.27, 0.45) for _ in range(n_runs-1)]
    light_poc_col.insert(0, light_poc)
    light_doc_col = [random.uniform(.022, 0.036) for _ in range(n_runs-1)]
    light_doc_col.insert(0, light_doc)
    light_water_col = [random.uniform(.19, .27) for _ in range(n_runs-1)]
    light_water_col.insert(0, light_water)
    
    params = pd.DataFrame({"run":list(range(1,n_runs+1)),
                           "p_max":p_max_col,
                           "IP":IP_col,
                           "theta_npp":theta_npp_col,
                           "theta_r":theta_r_col,
                           "sed_sink":sed_sink_col,
                           "resp_docl":resp_docl_col,
                           "resp_docr":resp_docr_col,
                           "resp_poc":resp_poc_col,
                           "settling_rate":settling_rate_col,
                           "sediment_rate":sediment_rate_col,
                           "light_poc":light_poc_col,
                           "light_doc":light_doc_col,
                           "light_water":light_water_col})
    params.to_csv("Peter Parameterization/results.csv", index = False)
    
    del params, p_max, IP, theta_npp, theta_r, sed_sink, resp_docl,resp_docr,resp_poc, settling_rate, sediment_rate, light_poc, light_doc,light_water

# model run
params = pd.read_csv("Peter Parameterization/results.csv")
total_runs=len(params)

while len(next(os.walk('Peter Parameterization/outputs'))[1]) <= total_runs:
    looptimestart = datetime.datetime.now()
    i = len(next(os.walk('Peter Parameterization/outputs'))[1])
    print("Commencing Run " + str(i+1))
    
    if i >= total_runs:
        print("All parameter sets have been processed.")
        break
    
    params = pd.read_csv("Peter Parameterization/results.csv")
    #p_max = 1 #/86400 #0.5 - 5
   # IP = 3e-5 #/86400 #0.1, 3e-5 #1e-5, 6e-5
    #theta_npp = 1.08 #1.08 #1.04-1.2
    #theta_r = 1.08 #1.08 #see theta npp
    #k_half = 0.5 #0.1-0.5
    #resp_docr = 0.001 #/ 86400 # 0.005-0.0001
    #resp_docl = 0.01 #/ 86400 # 0.01-0.1
    #sediment_rate = 0.1 #/ 86400 #0.1-1
    #piston_velocity = 1.0 #/ 86400 #not used
    #light_water = 0.125 #keep constant
    #light_doc = 0.02 #.01-.04 #maybe manually fit
    #light_poc = 0.7 #.2-1
    
 
        
    # sw_factor = params[i, "sw_factor"]
    # wind_factor = params[i, "wind_factor"]
    # at_factor = params[i, "at_factor"]
    # turb_factor = params[i, "turb_factor"]
    p_max = params.iloc[i, params.columns.get_loc("p_max")]
    IP = params.iloc[i, params.columns.get_loc("IP")]
    theta_npp = params.iloc[i, params.columns.get_loc("theta_npp")]
    theta_r = params.iloc[i, params.columns.get_loc("theta_r")]
    sed_sink = params.iloc[i, params.columns.get_loc("sed_sink")]
    # k_half = params[i, "k_half"]
    resp_docr = params.iloc[i, params.columns.get_loc("resp_docr")]
    resp_docl = params.iloc[i, params.columns.get_loc("resp_docl")]
    resp_poc = params.iloc[i, params.columns.get_loc("resp_poc")]
    settling_rate = params.iloc[i, params.columns.get_loc("settling_rate")]
    sediment_rate = params.iloc[i, params.columns.get_loc("sediment_rate")]
    # piston_velocity = params[i, "piston_velocity"]
    light_water = params.iloc[i, params.columns.get_loc("light_water")]
    light_doc = params.iloc[i, params.columns.get_loc("light_doc")]
    light_poc = params.iloc[i, params.columns.get_loc("light_poc")]
    
    del params
    
    res = run_wq_model(  
        u = deepcopy(u_ini),
        o2 = deepcopy(wq_ini[0]),
        docr = deepcopy(wq_ini[1]),
        docl = 1.0 * volume,
        pocr = 0.5 * volume,
        pocl = 0.5 * volume,
        startTime = startTime, 
        endTime = endTime, 
        area = area,
        volume = volume,
        depth = depth,
        zmax = zmax,
        nx = nx,
        dt = dt,
        dx = dx,
        daily_meteo = meteo_all[0],
        secview = meteo_all[1],
        phosphorus_data = tp_boundary,
        ice = False,
        Hi = 0,
        Hs = 0,
        Hsi = 0,
        iceT = 6,
        supercooled = 0,
        diffusion_method = 'pacanowskiPhilander',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
        scheme ='implicit',
        km = 1.4 * 10**(-7), # 4 * 10**(-6), 
        k0 = 1 * 10**(-2),
        weight_kz = 0.5,
        kd_light = 0.6, 
        denThresh = 1e-2,
        albedo = 0.01,
        eps = 0.97,
        emissivity = 0.97,
        sigma = 5.67e-8,
        sw_factor = 1.0,
        wind_factor = 1.2,
        at_factor = 1.0,
        turb_factor = 1.0,
        p2 = 1,
        B = 0.61,
        g = 9.81,
        Cd = 0.0013, # momentum coeff (wind)
        meltP = 1,
        dt_iceon_avg = 0.8,
        Hgeo = 0.1, # geothermal heat 
        KEice = 0,
        Ice_min = 0.1,
        pgdl_mode = 'on',
        rho_snow = 250,
        p_max = p_max/86400,#1
        IP = IP/86400 ,#0.1, 3e-5
        theta_npp = theta_npp, #1.08
        theta_r = theta_r, #1.08
        conversion_constant = 1e-4,#0.1
        sed_sink = sed_sink / 86400, #0.01
        k_half = 0.5,
        resp_docr = resp_docr/86400, # 0.001 0.0001
        resp_docl = resp_docl/86400, # 0.01 0.05
        resp_poc = resp_poc/86400, # 0.1 0.001 0.0001
        settling_rate = settling_rate/86400, #0.3
        sediment_rate = sediment_rate/86400,
        piston_velocity = 1/86400,
        light_water = light_water,
        light_doc = light_doc,
        light_poc = light_poc,
        mean_depth = sum(volume)/max(area),
        oc_load_input = .448  * max(area)*24, # 37.3 mmol C m-2d-1=.448gC m-2 d-1 (Cole et al,2006) divided by 24 hr/d
        hydro_res_time_hr = 1285*24 , #from Cole and Pace (1998)
        #outflow_depth = .01, #take out for Peter was out before and now jsut low
        prop_oc_docr = 0.762, #0.8 inflow
        prop_oc_docl = 0.058, #0.05 inflow
        prop_oc_pocr = 0.06, #0.05 inflow
        prop_oc_pocl = 0.12, #0.1 inflow
        W_str = None)

    temp=  res['temp']
    o2=  res['o2']
    docr=  res['docr']
    docl =  res['docl']
    pocr=  res['pocr']
    pocl=  res['pocl']
    # diff =  res['diff']
    # avgtemp = res['average'].values
    # temp_initial =  res['temp_initial']
    # temp_heat=  res['temp_heat']
    # temp_diff=  res['temp_diff']
    # temp_mix =  res['temp_mix']
    # temp_conv =  res['temp_conv']
    # temp_ice=  res['temp_ice']
    # meteo=  res['meteo_input']
    # buoyancy = res['buoyancy']
    # icethickness= res['icethickness']
    # snowthickness= res['snowthickness']
    # snowicethickness= res['snowicethickness']
    # npp = res['npp']
    # docr_respiration = res['docr_respiration']
    # docl_respiration = res['docl_respiration']
    # poc_respiration = res['poc_respiration']
    # kd = res['kd_light']
    secchi = res['secchi']
    # thermo_dep = res['thermo_dep']
    # energy_ratio = res['energy_ratio']
    
    
    doc_all = np.add(docl, docr)
    poc_all = np.add(pocl, pocr)
    
    print("temp shape:", temp.shape)
    print("temp dtype:", temp.dtype)
    print("temp values (first row):", temp[0, :10])
    print("Non-NaN per column:", np.sum(~np.isnan(temp), axis=0))
    print("Total non-NaN values:", np.sum(~np.isnan(temp)))
        
    run_folder = f"Peter Parameterization/outputs/Run_{i+1}"
    os.makedirs(run_folder, exist_ok=True)

# Convert depth to rounded labels for index
    depth_labels = [round(d, 2) for d in depth]

# Save each variable with depth as index, time as columns
    
    pd.DataFrame(temp,index=depth_labels, columns=times).to_csv(f"{run_folder}/templab.csv")
    pd.DataFrame(o2/ volume, index=depth_labels, columns=times).to_csv(f"{run_folder}/dolab.csv")
    pd.DataFrame(doc_all/volume, index=depth_labels, columns=times).to_csv(f"{run_folder}/doclab.csv")
    pd.DataFrame(poc_all/volume, index=depth_labels, columns=times).to_csv(f"{run_folder}/poclab.csv")
    pd.DataFrame(secchi).to_csv(f"{run_folder}/secchi.csv", index = False)
   
   # o2_conc = pd.DataFrame(o2 / volume[:, np.newaxis], index=depth_labels, columns=times)
   # doc_conc = pd.DataFrame(doc_all / volume[:, np.newaxis], index=depth_labels, columns=times)
   # poc_conc = pd.DataFrame(poc_all / volume[:, np.newaxis], index=depth_labels, columns=times)

# Save the concentration versions
   # o2_conc.to_csv(f"{run_folder}/do_mgL.csv")
   # doc_conc.to_csv(f"{run_folder}/doc_mgL.csv")
   # poc_conc.to_csv(f"{run_folder}/poc_mgL.csv")
    
    pd.DataFrame(temp.T).to_csv(f"{run_folder}/temp.csv", header=False, index=False)
    secchi = np.squeeze(secchi).flatten()

# Create a 2D array by repeating secchi across depths
    secchi_2d = np.tile(secchi[:, np.newaxis], (1, len(depth_labels)))  # shape (n_times, n_depths)

    pd.DataFrame(secchi_2d).to_csv(f"Peter Parameterization/outputs/Run_{i+1}/secchi.csv", index=False, header=False)
    pd.DataFrame((o2 / volume[:, np.newaxis]).T).to_csv(f"{run_folder}/do_mgL.csv", index=False, header=False)
    pd.DataFrame((doc_all / volume[:, np.newaxis]).T).to_csv(f"{run_folder}/doc_mgL.csv", index=False, header=False)
    pd.DataFrame((poc_all / volume[:, np.newaxis]).T).to_csv(f"{run_folder}/poc.csv", index=False, header=False)
    #save_long_format(poc_conc, 'poc', f"{run_folder}/poc_long.csv")
 #save withouth labels, still not working 
    #pd.DataFrame(temp).to_csv(f"{run_folder}/temp.csv",index= False)
    #pd.DataFrame(o2).to_csv(f"{run_folder}/do.csv",index= False)
    #pd.DataFrame(doc_all).to_csv(f"{run_folder}/doc.csv",index= False)
    #pd.DataFrame(poc_all).to_csv(f"{run_folder}/poc.csv",index= False)
    #pd.DataFrame(secchi).to_csv(f"{run_folder}/secchi.csv", index = False)
    
## trying to save a different way, this was the original, still had issues saving
   # os.mkdir("Peter Parameterization/outputs/Run_"+str(i+1))
    #pd.DataFrame(temp).to_csv("Peter Parameterization/outputs/Run_"+str(i+1)+"/temp.csv", index = False)
    #pd.DataFrame(o2).to_csv("Peter Parameterization/outputs/Run_"+str(i+1)+"/do.csv", index = False)
    #pd.DataFrame(doc_all).to_csv("Peter Parameterization/outputs/Run_"+str(i+1)+"/doc.csv", index = False)
    #pd.DataFrame(poc_all).to_csv("Peter Parameterization/outputs/Run_"+str(i+1)+"/poc.csv", index = False)
    #pd.DataFrame(secchi).to_csv("Peter Parameterization/outputs/Run_"+str(i+1)+"/secchi.csv", index = False)
    
    del p_max, IP, theta_npp, theta_r, sed_sink, resp_docl,resp_docr, resp_poc, settling_rate, sediment_rate, light_poc, light_water, light_doc,temp, o2, docr, docl, doc_all, pocr, pocl, poc_all, secchi
    
    print("Run " + str(i+1) + " finished at " + str(datetime.datetime.now()))
    looptimeend = datetime.datetime.now()
    
    looplengthdiff = looptimeend - looptimestart
    # if looplengthdiff.seconds/60/60 >= 1:
    #     sys.stdout.flush()
    #     os._exit(00)
    
    del i, looplengthdiff, looptimeend, looptimestart
    gc.collect()
        



#End = datetime.datetime.now()
#print(End - Start)
