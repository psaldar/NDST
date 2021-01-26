# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:49:42 2020

@author: Pablo Saldarriaga
"""
### script for simulating data for dynamic segmentation
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
### set random seed to replicate experiments
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
#%%
### Simulation parameters
num_groups = 3  ### number of groups
num_vars = 10  ### number of variables per individual
val = 8  ### range to initialize centroids
periods = 15  ### number of periods to simulate
perc2keep = 0.2  ### Percentage of individuals to keep each period
N = 600  ### Number of individuals per period
const_corr = 5  ### constant to add correlation to variables
trend = 1  ### trend for random walk on centroids

### parameters to include case-wise outliers
outliers = False
cant_outliers = int(N * 0.09) ### 0.01 to 0.09

### parameters to include cell-wise outliers
outliers_cellwise = False  ### the increase in this scenario is 15
outliers_cellwise_h = True
increase_std = 10
porc_outliers = 0.09  ### 0.01 to 0.09

### parameter to visualize principal components of the simulated dataset, if
### false, any plot will show up
plot = True
#%%
### initialize groups centroids

### centroid for each group will be a random vector in which each component
### takes a value between (-val, val)
mus = []

for i in range(num_groups):
    centroid = []
    for j in range(num_vars):
        c_ij = random.choice([-1, 1]) * np.random.random() * val
        centroid.append(c_ij)
    mus.append(centroid)

### move centroids of each group using a random walk
centroids_per_period = []
for i in range(num_groups):
    mu = np.array(mus[i])
    mu_periods = []
    for _ in range(periods):

        next_mu = mu + np.random.normal(size=num_vars) + trend
        mu_periods.append(next_mu)

        mu = next_mu
    centroids_per_period.append(mu_periods)
#%%
### change standar deviation for each variable
sigmas = []
for i in range(num_groups):
    sigma = []
    for j in range(num_vars):
        sigma_ij = np.random.random() * 2
        sigma.append(sigma_ij)
    sigmas.append(sigma)

### move standard deviation of each variable in each group using a random walk
sigmas_per_period = []
for i in range(num_groups):
    sigma_i = np.array(sigmas[i])
    sigma_periods = []
    for _ in range(periods):

        ### we put abs on the random walk to avoid negative values
        next_sigma = abs(sigma_i + np.random.normal(size=num_vars))
        sigma_periods.append(next_sigma)

        sigma_i = next_sigma
    sigmas_per_period.append(sigma_periods)

### if cell-wise outliers are added, then for each group and each period
### some random components will change its behavior (change its standar deviation)
sigmas_per_period_outlier = []
if outliers_cellwise:
    for i in range(num_groups):
        sigma_periods_out = []
        for j in range(periods):
            ### select variables that will be changed for adding cell-wise
            ### outliers.
            num_vars_out = random.randint(1, int(num_vars / 2))  ### random.randint(1, 2)
            vars_out_period = np.argsort(np.random.random(num_vars))[:num_vars_out]

            ### change the standar deviation of the vth variable if it appears
            ### in vars_out_period, otherwise, keep the original standar deviation
            res_vars = []
            for v in range(num_vars):
                if v in vars_out_period:
                    res_vars.append(sigmas_per_period[i][j][v] * increase_std)
                else:
                    res_vars.append(sigmas_per_period[i][j][v].copy())

            sigma_periods_out.append(np.array(res_vars))

        ### Here the new sigma is created for adding outliers during simulation
        sigmas_per_period_outlier.append(sigma_periods_out)

#%%
### generate number of records per group per period
low_v = int(N / (2 * num_groups))
upp_v = int(N / (num_groups))

hist_n_obs = []
for j in range(periods):
    n_obs = []
    for i in range(num_groups):

        if i == num_groups - 1:
            n_obs.append(N - sum(n_obs))

        else:
            n_obs.append(random.randint(low_v, upp_v))
    hist_n_obs.append(n_obs)

### groups creation

### random selection of variables with low importance
ones = np.ones((1, num_vars))
vars2change = 3
change = random.sample(range(num_vars), vars2change)

### if the position vth is 1, then the vth variable is relevant
### otherwise it is not
for v in change:
    ones[0][v] = 0

    ### remove importance on selected variables
    for i in range(num_groups):
        for j in range(periods):
            sigmas_per_period[i][j][v] = 1
            centroids_per_period[i][j][v] = 0

partial_results = []
for i in range(num_groups):
    mus_group = centroids_per_period[i]
    sigmas_group = sigmas_per_period[i]

    prev_group = pd.DataFrame()
    for j in range(periods):

        ### In the first period, all individuals are generated, then in next
        ### periods, there will be some individuals from last period and new ones
        ### to be generated
        if j == 0:
            n = hist_n_obs[j][i]
        else:
            n_new = len(prev_group)
            n = hist_n_obs[j][i] - n_new

        ### factor to add correlation between variables
        fact = ones * (np.random.random(size=(n, 1)) * const_corr)

        if outliers_cellwise_h:

            data = np.random.normal(size=(n, num_vars)) * sigmas_group[j] + mus_group[j] + fact

            ### select cells to be contaminated
            cellOut = (np.random.random(size=(n, num_vars)) <= porc_outliers).astype(int)

            ### add cell-wise outliers to dataset
            data = data + cellOut * (increase_std * sigmas_group[j])

        elif outliers_cellwise:
            sigmas_group_out = sigmas_per_period_outlier[i]

            if j == 0:
                n_outliers_sim = int(hist_n_obs[j][i] * porc_outliers)
                n_sim_period = hist_n_obs[j][i] - n_outliers_sim
            else:
                n_outliers_sim = int(hist_n_obs[j][i] * porc_outliers)
                n_sim_period = hist_n_obs[j][i] - n_new - n_outliers_sim

            ### data simulation
            fact = ones * (np.random.random(size=(n_sim_period, 1)) * const_corr)
            ### simulate data with the defined mean and variance
            data_sim_per = np.random.normal(size=(n_sim_period, num_vars)) * sigmas_group[j] + mus_group[j] + fact

            ### Outlier simulation
            fact = ones * (np.random.random(size=(n_outliers_sim, 1)) * const_corr)
            ### add cell-wise outliers, simulating data but changing some of the
            ### standar deviations of some components
            outliers_sim = np.random.normal(size=(n_outliers_sim, num_vars)) * sigmas_group_out[j] + mus_group[j] + fact

            data = np.append(data_sim_per, outliers_sim, axis=0)
        else:
            ### if no cell-wise outliers will be added, then generate all n
            ### individuals
            data = np.random.normal(size=(n, num_vars)) * sigmas_group[j] + mus_group[j] + fact


        ############# Optional modifications for the simulation ###############
        ### IF we wanted to easily add additional dispersion in the groups, we could activate this part
        ### (This could be useful to have a different simulation scenario)
        ### (Add the same noise for all the variables, not only for the important ones)
        additional_dispersion = False
        auxili = np.random.normal(size=(n, num_vars))*0.05 ### Additional noise
        if additional_dispersion:
            data = data+auxili      
            
        ### We could also add another factor to increase correlation in all variables
        ### (By group, but for all variables)
        factor_for_all = False
        c_ij = random.choice([-1, 1]) * np.random.random() * val * 0.1
        if factor_for_all:
            data = data+c_ij
        #######################################################################

        ### Save as dataframe
        aux = pd.DataFrame()
        aux['country'] = range(n)
        aux['country'] = f'g_{i}_{j}_' + aux['country'].astype(str)
        aux['Date'] = j
        aux['group'] = i

        for k in range(num_vars):
            aux[f'X_{k}'] = data[:, k]

        aux = pd.concat([aux, prev_group], ignore_index=True)
        aux['Date'] = j

        ### keep a proportion of individuals to be considered in the next period
        prev_group = aux.sample(frac=perc2keep).reset_index(drop=True).copy()

        partial_results.append(aux)

### include outliers in each period (cell-wise outliers)
if outliers:
    random.seed(SEED)
    np.random.seed(SEED)
    for j in range(periods):
        ### for each period, an outlier group is simulated and added to the
        ### data
        mus_group = ((np.random.random((1, num_vars)) - 0.5) * 2)[0]
        
        
        s_outlier = (np.random.random((1, num_vars)) * 10)[0]

        ### Factor to affect the groups mean making it move over all space
        facto_mug = 50

        data = np.random.normal(size=(cant_outliers, num_vars)) * s_outlier + mus_group * facto_mug
        print(mus_group)
        aux = pd.DataFrame()
        aux['country'] = range(cant_outliers)
        aux['country'] = f'out_{i}_{j}_' + aux['country'].astype(str)
        aux['Date'] = j
        aux['group'] = -1

        for k in range(num_vars):
            aux[f'X_{k}'] = data[:, k]

        partial_results.append(aux)

sim_data = pd.concat(partial_results, ignore_index=True)

### Shuffle the dataset to have individuals from groups mixed, since at this point
### all data per period is sorted by group
sim_data = sim_data.sample(frac=1).sort_values(by='Date').reset_index(drop=True)

if not os.path.exists('../data'):
    os.mkdir('../data')

### save the simulated dataset
sim_data.to_csv('../data/simulated_data.csv', sep=',', index=False)
#%%
### plot data using PCA
if plot:
    ### adjust PCA on the first period
    data_period = sim_data.reset_index(drop=True)
    
    scaler = StandardScaler()
    scaler.fit(data_period[data_period.columns[3:]])
    
    pca = PCA(n_components=2)
    pca.fit(scaler.transform(data_period[data_period.columns[3:]]))
    
    ### set colors of the groups
    colores = ['purple','green',   'blue','yellow']
    
    figu = plt.figure()
    import matplotlib
    for p in range(min(periods, 15)):
        data_period = sim_data[sim_data.Date == p].reset_index(drop=True)
    
        z = pca.transform(scaler.transform(data_period[data_period.columns[3:]]))
    
        ax = plt.subplot(5, 3, p + 1)
        plt.scatter(z[:, 0], z[:, 1], c=data_period.group, cmap=matplotlib.colors.ListedColormap(colores))
        plt.title('Period ' + str(p + 1))
        figu.tight_layout(pad=0.1)

    if not os.path.exists('../data/outputs'):
        os.mkdir('../data/outputs')
        
    ### Save pca scaler
    import pickle
    with open('../data/outputs/pca_scaler.pickle', 'wb') as f:
        pickle.dump(pca, f)