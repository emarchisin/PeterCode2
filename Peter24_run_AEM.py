import numpy as np
import pandas as pd
import os
#import scipy
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
#os.chdir("D:/bensd/Documents/Python_Workspace/1D-AEMpy/src")
os.chdir("/Users/emmamarchisin/Desktop/Research/Code/Cascade Lakes/Peter Lake/PeterCode2")
from Peter24_processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 19 # maximum lake depth
nx = 19 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = 'Peter Inputs/2022/peter_bath.csv',
                            dx = dx, nx = nx)
                            
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = 'Peter Inputs/2022/nldas_hourly_Cascade2224.csv',
                    secchifile = None, 
                    windfactor = 0.5) #1.0

pd.DataFrame(meteo_all[0]).to_csv("Peter Inputs/2022/meteorology_input2.csv", index = False)
                     
## time step discretization 
#n_years = 0.05#0.307
#hydrodynamic_timestep = 24 * dt
#total_runtime =  (365 * n_years) * hydrodynamic_timestep/dt  
#startTime =  1695 # DOY in 2016 * 24 hours- adjust for PYthon starting on 0 index
#endTime =  4381# * hydrodynamic_timestep/dt) - 1
#burn in with 2022
n_years = 2.32#0.307
hydrodynamic_timestep = 24 * dt
total_runtime =  (365 * n_years) * hydrodynamic_timestep/dt  
startTime =  3759 # DOY in 2016 * 24 hours- adjust for PYthon starting on 0 index, basically inced of start date
endTime =  24104# * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)

## here we define our initial profile
u_ini = initial_profile(initfile = 'Peter Inputs/2022/peter_wtemph.csv', nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate) 

wq_ini = wq_initial_profile(initfile = 'Peter Inputs/2022/peter_driver2.csv', nx = nx, dx = dx,
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)

tp_boundary = provide_phosphorus(tpfile ='Peter Inputs/2022/peter_tp2.csv', 
                                 startingDate = startingDate,
                                 startTime = startTime)

tp_boundary = tp_boundary.dropna(subset=['observation'])

Start = datetime.datetime.now()

    
res = run_wq_model(  
    u = deepcopy(u_ini),
    o2 = deepcopy(wq_ini[0]),
    docr = deepcopy(wq_ini[1]) * 1.3,
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
    wind_factor = 0.5, #1.2, #1.2
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
    p_max = 3/86400,#1
    IP = 3e-5/86400 ,#0.1, 3e-5
    theta_npp = 1.08, #1.08
    theta_r = 1.08, #1.08 #1.5 for 104 #1.35 for 106
    conversion_constant = 1e-4,#0.1
    sed_sink = -10/ 86400, #0.01 #-.12 , 0.0626
    k_half = 0.5,
    resp_docr =  0.0001/86400, # 0.001 0.0001 (s-1) 0.007/86400
    resp_docl =0.05/86400 , # 0.01 0.05 (s-1) 0.05/86400
    resp_poc = 0.1/86400, # 0.1 0.001 0.0001 (s-1_)
    settling_rate = 0.1/86400, #0.3 (ms-1)
    sediment_rate = 0.7/86400, #(m s-1)
    piston_velocity = 1.0/86400,
    light_water = 0.3, #(m-1) #0.3
    light_doc = 0.02, #(m-1)
    light_poc = 0.5,
    oc_load_input =(.456)  * max(area)/24, # 38.0 mmol C m-2d-1=.456gC m-2 d-1 (Cole et al,2006 DIF model) divided by 24 hr/d
    hydro_res_time_hr = 365*24,#1285?
    outflow_depth = 0.5,
    prop_oc_docr = 0.762, #0.8 inflow, 0.747, 0.573, 0.4
    prop_oc_docl = 0.058, #0.05 inflow,  0.058, 0.245, 
    prop_oc_pocr = 0.06, #0.05 inflow, .075, 0.081, 0.167
    prop_oc_pocl = 0.12, #0.1 inflow, .12
    mean_depth = sum(volume)/max(area),
    W_str = None,
   # training_data_path = 'Peter Parameterization/outputs', #'../output' for putting into a ML model
    timelabels = times)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']
npp = res['npp']
docr_respiration = res['docr_respiration']
docl_respiration = res['docl_respiration']
poc_respiration = res['poc_respiration']
kd = res['kd_light']
secchi = res['secchi']
thermo_dep = res['thermo_dep']
energy_ratio = res['energy_ratio']


End = datetime.datetime.now()
print(End - Start)

doc_all = np.add(docl, docr)
poc_all = np.add(pocl, pocr)

""" plt.figure(figsize=(10, 5))
plt.plot(times, energy_ratio[0,:])
plt.ylabel("Energy Ratio", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.show()

print(times)
print(thermo_dep[0,:])
plt.figure(figsize=(10, 5))
plt.plot(times, thermo_dep[0,:]/2)
plt.ylabel("Thermocline depth", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.show()
 """

df_obs = pd.read_csv('Peter Inputs/Obs4check/observed_data3.csv',  parse_dates=True)
df_obs_surf = df_obs[(df_obs['variable'] == 'wtemp') & (df_obs['depth'] == 1)]
df_obs_surf = df_obs_surf.drop(columns=['depth_id'])
df_obs_surf = df_obs_surf.dropna(subset=['observation'])
print(df_obs_surf.head())
print(df_obs_surf.shape)
print(df_obs_surf.columns)
print(df_obs_surf["datetime"])
print( df_obs_surf["observation"])
df_obs_surf['datetime'] = pd.to_datetime(df_obs_surf['datetime'], format = 'mixed')

df_obs_bot= df_obs[(df_obs['variable'] == 'wtemp') & (df_obs['depth'] == 8)]
df_obs_bot = df_obs_bot.drop(columns=['depth_id'])
#df_obs_bot = df_obs_bot.dropna()
df_obs_bot['datetime'] = pd.to_datetime(df_obs_bot['datetime'], format = 'mixed')

print(u_ini)
print(depth)
plt.figure(figsize=(10, 5))
plt.plot(times, temp[2,:], color= 'blue', label='1m Modeled Temp', linestyle= 'solid')
plt.plot(times, temp[16,:], color= 'blue', label='8m Modeled Temp', linestyle= 'dashed')
plt.plot(df_obs_surf["datetime"], df_obs_surf["observation"], color= 'red', label='1m Observed Temp', linestyle= 'solid')
plt.plot(df_obs_bot["datetime"], df_obs_bot["observation"], color= 'red', label='8m Observed Temp', linestyle= 'dashed')
plt.ylabel("Temp.", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.legend(loc='best')
plt.show()

df_obs_surf_do = df_obs[(df_obs['variable'] == 'do_mgl') & (df_obs['depth'] == 1)]
df_obs_surf_do = df_obs_surf_do.drop(columns=['depth_id'])
#df_obs_surf_do = df_obs_surf.dropna()
print(df_obs_surf_do.head())
print(df_obs_surf_do.shape)
print(df_obs_surf_do.columns)
print(df_obs_surf_do["datetime"])
print( df_obs_surf_do["observation"])
df_obs_surf_do['datetime'] = pd.to_datetime(df_obs_surf_do['datetime'], format = 'mixed')

df_obs_bot_do= df_obs[(df_obs['variable'] == 'do_mgl') & (df_obs['depth'] == 8)]
df_obs_bot_do = df_obs_bot_do.drop(columns=['depth_id'])
df_obs_bot_do = df_obs_bot_do.dropna(subset=['observation'])
df_obs_bot_do['datetime'] = pd.to_datetime(df_obs_bot_do['datetime'], format = 'mixed')

print(u_ini)
print(depth)
plt.figure(figsize=(10, 5))
plt.plot(times, o2[2,:]/volume[2], color= 'blue', label='1m Modeled DO', linestyle= 'solid')
plt.plot(times, o2[16,:]/volume[16], color= 'blue', label='8m Modeled DO', linestyle= 'dashed')
plt.plot(df_obs_surf_do["datetime"], df_obs_surf_do["observation"], color= 'red', label='1m Observed DO', linestyle= 'solid')
plt.plot(df_obs_bot_do["datetime"], df_obs_bot_do["observation"], color= 'red', label='8m Observed DO', linestyle= 'dashed')
plt.ylabel("DO.", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.legend(loc='best')
plt.show()

df_obs_surf_doc = df_obs[(df_obs['variable'] == 'doc') & (df_obs['depth'] == 1)]
df_obs_surf_doc = df_obs_surf_doc.drop(columns=['depth_id'])
df_obs_surf_doc['datetime'] = pd.to_datetime(df_obs_surf_doc['datetime'], format = 'mixed')

df_obs_bot_doc= df_obs[(df_obs['variable'] == 'doc') & (df_obs['depth'] == 12)]
df_obs_bot_doc = df_obs_bot_doc.drop(columns=['depth_id'])
df_obs_bot_doc = df_obs_bot_doc.dropna(subset=['observation'])
df_obs_bot_doc['datetime'] = pd.to_datetime(df_obs_bot_doc['datetime'], format = 'mixed')

plt.figure(figsize=(10, 5))
plt.plot(times, doc_all[2,:]/volume[2], color='blue', label='1m Modeled DOC', linestyle='solid')
plt.plot(times, doc_all[16,:]/volume[16], color='blue', label='12m Modeled DOC', linestyle='dashed')
plt.scatter(df_obs_surf_doc["datetime"], df_obs_surf_doc["observation"], 
            color='red', label='1m Observed DOC', marker='o', zorder=5)
plt.scatter(df_obs_bot_doc["datetime"], df_obs_bot_doc["observation"], 
            color='red', label='12m Observed DOC', marker='x', zorder=5)
plt.ylabel("DOC (mg/L)", fontsize=15)
plt.xlabel("Time", fontsize=15)
plt.legend(loc='best')
plt.show()

# heatmap of temps  
N_pts = 30 #how many points on x axis

#quit() # was cut out before
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),yticklabels=2, vmin = 0, vmax = 30), #xticklabels=1000,
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  (dC)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()




fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(o2)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  yticklabels=2, vmin = 0, vmax = 20)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Dissolved Oxygen  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts* n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 15)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 15)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(npp)/volume) * 86400, cmap=plt.cm.get_cmap('Spectral_r'), yticklabels=2, vmin = 0, vmax = .3)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("NPP  (g/m3/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docr_respiration , cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 2e-3)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCr respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docl_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  yticklabels=2, vmin = 0, vmax = 8e-2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCl respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years)) #
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(poc_respiration , cmap=plt.cm.get_cmap('Spectral_r'),   yticklabels=2, vmin = 0, vmax = 3e-1)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label, rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

# plt.plot(npp[1,1:400]/volume[1] * 86400)
# plt.plot(o2[1,:]/volume[1])
# plt.plot(o2[1,1:(24*14)]/volume[1])
# plt.plot(o2[1,:]/volume[1])
# plt.plot(docl[1,:]/volume[1])
# plt.plot(docr[1,1:(24*10)]/volume[1])
# plt.plot(pocl[0,:]/volume[0])
# plt.plot(pocr[0,:]/volume[0])
# plt.plot(npp[0,:]/volume[0]*86400)
# plt.plot(docl_respiration[0,:]/volume[0]*86400)
# plt.plot(o2[(nx-1),:]/volume[(nx-1)])

plt.plot(o2[1,1:(24*28)]/volume[1]/4, color = 'blue', label = 'O2')
gpp = npp[1,:] -1/86400 *(docl[1,:] * docl_respiration[1,:]+ docr[1,:] * docr_respiration[1,:] + pocl[1,:] * poc_respiration[1,:] + pocr[1,:] * poc_respiration[1,:])
plt.plot(npp[1,1:(24*28)]/volume[1] * 86400, color = 'yellow', label = 'NPP') 
plt.plot(1/86400*(docl[1,1:(24*28)] * docl_respiration[1,1:(24*28)]+ docr[1,1:(24*28)] * docr_respiration[1,1:(24*28)] + pocl[1,1:(24*28)] * poc_respiration[1,1:(24*28)] + pocr[1,1:(24*28)] * poc_respiration[1,1:(24*28)])/volume[1] * 86400, color = 'red', label = 'R') 
plt.plot(gpp[1:(24*28)]/volume[1] * 86400, color = 'green', label = 'GPP')
plt.legend(loc='best')
plt.show() 

plt.plot(times, kd[0,:])
plt.ylabel("kd (/m)")
plt.show()

plt.plot(times, secchi[0,:])
plt.ylabel("Secchi Depth (m)")
plt.xlabel("Time")
plt.show()

do_sat = o2[0,:] * 0.0
for r in range(0, len(temp[0,:])):
    do_sat[r] = do_sat_calc(temp[0,r], 982.2, altitude = 516) 

plt.plot(times, o2[0,:]/volume[0], color = 'blue')
plt.plot(times, do_sat, color = 'red')
plt.ylabel("DO (mg/L)")
plt.xlabel("Time")
plt.legend()
plt.show()

plt.plot(times, thermo_dep[0,:]*dx,color= 'blue')
plt.plot(times, temp[0,:] - temp[(nx-1),:], color = 'red')
plt.show()

depths = [1,34]   # Python indices for depth=1 and depth=12
labels = ['Depth 1', 'Depth 17']

# Plot DO
plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    plt.plot(times, o2[d, :] / volume[d], label=f'DO at {labels[i]}', linestyle='-', color=('blue' if d == 1 else 'cyan'))
plt.ylabel("DO (mg/L)")
plt.xlabel("Time")
plt.legend()
plt.title("Dissolved Oxygen (DO)")
plt.show()

plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    plt.plot(times, doc_all[d, :]/volume [d], label=f'DOC at {labels[i]}', linestyle='-', color=('green' if d == 1 else 'lightgreen'))
plt.ylabel("DOC (mg/L)")
plt.xlabel("Time")
plt.legend()
plt.title("Dissolved Organic Carbon (DOC)")
plt.show()

# Plot POC
plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    plt.plot(times, poc_all[d, :]/volume[d], label=f'POC at {labels[i]}', linestyle='-', color=('orange' if d == 1 else 'gold'))
plt.ylabel("POC (mg/L)")
plt.xlabel("Time")
plt.legend()
plt.title("Particulate Organic Carbon (POC)")
plt.show()

# TODO
# air water exchange
# sediment loss POC
# diffusive transport
# r and npp
# phosphorus bc
# ice npp
# wind mixingS

pd.DataFrame(temp).to_csv("Peter Parameterization/Test/PT.modeled_temp.csv")
pd.DataFrame(o2).to_csv("Peter Parameterization/Test/PT.modeled_do.csv")
pd.DataFrame(docr).to_csv("Peter Parameterization/Test/PT.modeled_docr.csv")
pd.DataFrame(docl).to_csv("Peter Parameterization/Test/PT.modeled_docl.csv")
pd.DataFrame(pocl).to_csv("Peter Parameterization/Test/PT.modeled_pocl.csv")
pd.DataFrame(pocr).to_csv("Peter Parameterization/Test/PT.modeled_pocr.csv")
pd.DataFrame(secchi).to_csv("Peter Parameterization/Test/PT.modeled_secchi.csv")
pd.DataFrame(thermo_dep).to_csv("Peter Parameterization/Test/PT.modeled_thermo_dep.csv")
#pd.DataFrame(temp,index=depth, columns=times).to_csv("Peter Parameterization/Test/templab.csv")
#pd.DataFrame(o2/ volume, index=depth, columns=times).to_csv("Peter Parameterization/Test/dolab.csv")
#pd.DataFrame(doc_all/volume, index=depth, columns=times).to_csv("Peter Parameterization/Test/doclab.csv")
#pd.DataFrame(poc_all/volume, index=depth, columns=times).to_csv("Peter Parameterization/Test/poclab.csv")


# pd.DataFrame(temp).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_temp.csv")
# pd.DataFrame(o2).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_do.csv")
# pd.DataFrame(docr).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_docr.csv")
# pd.DataFrame(docl).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_docl.csv")
# pd.DataFrame(pocl).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_pocl.csv")
# pd.DataFrame(pocr).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_pocr.csv")
# pd.DataFrame(secchi).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_secchi.csv")


# label = 116
# doc_all = np.add(docl, docr)
# poc_all = np.add(pocl, pocr)
# os.mkdir("../parameterization/output/Run_"+str(label))
# pd.DataFrame(temp).to_csv("../parameterization/output/Run_"+str(label)+"/temp.csv", index = False)
# pd.DataFrame(o2).to_csv("../parameterization/output/Run_"+str(label)+"/do.csv", index = False)
# pd.DataFrame(doc_all).to_csv("../parameterization/output/Run_"+str(label)+"/doc.csv", index = False)
# pd.DataFrame(poc_all).to_csv("../parameterization/output/Run_"+str(label)+"/poc.csv", index = False)
# pd.DataFrame(secchi).to_csv("../parameterization/output/Run_"+str(label)+"/secchi.csv", index = False)
