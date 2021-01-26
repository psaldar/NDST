# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:38:34 2019

@author: nicol
"""
#%%
import os
import sys
import time
import functions
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
### Logger configuration
import logging
from logging.handlers import RotatingFileHandler

file_name = 'dynamic_segmentation'
logger = logging.getLogger()
dir_log = f'../logs/{file_name}.log'

### Create log folder if it does not exists
if not os.path.exists('../logs'):
    os.mkdir('../logs')

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
                    handlers=[handler])
#%%
logger.info('*' * 80)
logger.info('Start execution')

### Read data for gapminder experiments
#experiment = 3
#datos = pd.read_csv(f'../data/data_gapminder_experiment{experiment}.csv')

### Read data for simulated dataset
datos = pd.read_csv('../data/simulated_data.csv').drop(columns = 'group')

### Variables to be used
datos = datos[datos.columns[:]]

### Variables name
nomb_vars = datos.columns[2:]


### Data standarization
datos_e = datos.copy()
scaler_es = StandardScaler()
datos_e[datos_e.columns[2:]] = scaler_es.fit_transform(datos_e[datos_e.columns[2:]])
datos_e = datos_e.fillna(0)

### Applies PCA to all dataset
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_e[datos_e.columns[2:]])

### The following code is just for style purposes (invert sign of the first
### principal component, that way the more to the right the point is, the
### country that it represents is more developed)
de_gapminder = True
if de_gapminder:
    datos_pca[:, 0] = datos_pca[:, 0] * -1

### Fix random seed for replicability purposes
seed = 10

np.random.seed(seed)

##### Start keeping information from fisrt year
year_i = min(datos_e['Date'])  ### first year
filtro = datos_e['Date'] == year_i
X_data_df = datos_e[filtro].reset_index(drop=True)
this_indi = pd.unique(X_data_df.country)
tot_indiv = pd.unique(datos_e.country)
X_data = np.array(X_data_df[X_data_df.columns[2:]])

### Number of periods that will be included in this study (not including the
### first one)
periodos_incluir = max(datos_e['Date']) - year_i

### Filter PCA data
X_data_pca = np.array(datos_pca[filtro])

### list where labels assigned to each cluster are stored
etiquetas_glo = []

### list where per period and per cluster, the relevant variables are stored
imp_periods_var = []

### list where the centroids in each iteration are being stored
centroids_ite = []

### Number of observations in each period
numdata = len(X_data)

### Defines number of clusters (k), maximun number of iterations and distance
### to be used in the clustering technique
k = 3
numiter = 25
p_dista = 2

### Enable robust version
robust = False
if len(sys.argv) >= 5:
    robust = sys.argv[4]


### Centroids initialization
centroids = functions.init_centroids(X_data, k)

#%%
t_ini = time.time()

### Execute kmeans algorithm
etiquetas, centroids = functions.kmeans(X_data,
                                        numiter,
                                        centroids,
                                        p_dista=p_dista,
                                        indivs=this_indi,
                                        prev_i=[],
                                        robust=robust)


### Initialize previous centrois variable to be used in second phase of the
### technique
centroids_p = centroids.copy()

### Append labels
etiquetas_glo.append(etiquetas.copy())

### Append centroids
centroids_ite.append(centroids.copy())

### Global variable where variables importance in each iteration are stored
imp_iters = []

### Obtain variable importance for each cluster (highest to lowest)
importancias_cluster = []

### Iterate for each cluster
for clu in range(k):
    ### Keep information for each cluster
    datax_i = pd.DataFrame(X_data)
    datay_i = etiquetas.copy()
    
    ### Create a filter to identify those observations that do not belong to
    ### the cluster, then assign a -1 label to those observations. That way,
    ### there is a binary identification that says whether an observation
    ### belongs to that cluster ot not
    distintos_cluster = np.where(datay_i != clu)
    datay_i[distintos_cluster] = -1
    datay_i = pd.DataFrame(datay_i)

    ### Calculate variable relevances using a tree-based machine learning
    ### supervised method
    relevancias, _ = functions.variables_relevantes_arbol(datax_i, datay_i, 0)
    importancias_cluster.append(relevancias)

### Calculate for each variable, its average importance
imp_clus_prom = np.mean(importancias_cluster, axis=0)

### append importances in this iteration
imp_iters.append(imp_clus_prom)

### Append overall importances
imp_periods_var.append(importancias_cluster)

###############################################################################
################### Now, Start itrations for t >=2 ############################
###############################################################################

for periodos in range(periodos_incluir):

    ### Save previous X_data
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()

    ### Update data with the current period information
    prev_indi = this_indi.copy()
    X_data_df = datos_e[datos_e['Date'] == year_i + 1 + periodos].reset_index(drop=True)
    this_indi = pd.unique(X_data_df.country)
    X_data = np.array(X_data_df[X_data_df.columns[2:]])
    
    ### Obtain the first two principal components
    X_data_pca = np.array(datos_pca[datos_e['Date'] == year_i + 1 + periodos])

    ###########################################################################
    ########################### Dynamic weighting #############################
    ###########################################################################

    ### Weight X_data
    X_data_ori = X_data.copy()  ### X_data original (without weighting)
    
    ### Obtain average imporance for each variable (mean across all iterations)
    if robust:
        ### For the robust version, use the median instead of mean
        importancia_prom = np.median(imp_iters, axis=0)
    else:
        importancia_prom = np.mean(imp_iters, axis=0)

    ### Rank variable imporance from lowest to highest
    rank_variables = np.argsort(importancia_prom)
    rankpeso_variables = np.zeros(len(rank_variables))

    cont = 0
    for i in rank_variables:
        rankpeso_variables[i] = (cont + 1) / len(rank_variables)
        cont = cont + 1

    #### Use either ranking or average to weight
    peso_variables = importancia_prom.copy() * 100  ### Scale the weights to avoid numerical errors

    ### Weight X_data according to the importances
    X_data_pond = X_data.copy()
    for peso in range(len(peso_variables)):
        X_data_pond[:, peso] = X_data_pond[:, peso] * (peso_variables[peso] + 0.000001)


    ### Current label for each observation for each cluster
    etiquetas_prev = etiquetas.copy()
    if periodos == 0:
        etiquetas_prev_p = etiquetas.copy()

    ### Here, it is possible to use "args" in the script to execute different
    ### versions of the technique (without weightinh, without momentum
    ### without preserving centroids, etc.)
    
    if len(sys.argv) > 1:
        print('Usando argumentos de sys.argv...')
        con_centroids = sys.argv[1]
        con_ponderacion = sys.argv[2]
        con_momentum = sys.argv[3]

        if not con_ponderacion == 'True':
            X_data_pond = X_data_ori.copy()  ### If executed, data is not weighted
        if not con_centroids == 'True':
            centroids = functions.init_centroids(X_data_pond, k)  ### If executed, centroids are not keep for next period
            centroids_p = functions.init_centroids(X_data_ori, k)
        if not con_momentum == 'True':
            etiquetas = []  ### If executed, momentum is not used in the technique
            etiquetas_prev_p = []

    else:
        con_centroids = 'True'
        con_ponderacion = 'True'
        con_momentum = 'True'        

###########################################################################
################### Clusters using weighted k-means #######################
###########################################################################

    etiquetas, centroids = functions.kmeans(
                                            X_data_pond,
                                            numiter,
                                            centroids,
                                            p_dista=p_dista,
                                            etiquetas=etiquetas,
                                            indivs=this_indi,
                                            prev_i=prev_indi,
                                            robust=robust)

    ### Append labels
    etiquetas_glo.append(etiquetas.copy())
    
    ### Append centroids (using unweighted data to obtain the real cluster
    ### characteristics)
    centroids_ite.append(centroids.copy() * (1 / (peso_variables + 0.000001)))

    ### Correccion por si no se modificaron
    if len(centroids_ite) >= 2:
        uniqetis = pd.unique(etiquetas)
        for c in range(len(centroids)):
            if c not in uniqetis:
                centroids_ite[-1][c] = centroids_ite[-2][c]

    ### Obtain variables importance of each cluster (from highest to lowest)
    importancias_cluster = []
    
    ### Iterate over each cluster
    for clu in range(k):
        ### Keep information for each cluster
        datax_i = pd.DataFrame(X_data_pond)
        datay_i = etiquetas.copy()
        
        ### Create a filter to identify those observations that do not belong to
        ### the cluster, then assign a -1 label to those observations. That way,
        ### there is a binary identification that says whether an observation
        ### belongs to that cluster ot not
        distintos_cluster = np.where(datay_i != clu)

        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)

        ### Obtain relevances
        relevancias, _ = functions.variables_relevantes_arbol(datax_i, datay_i, 0)
        importancias_cluster.append(relevancias)
    
    ### Append overall imporances
    imp_periods_var.append(importancias_cluster)

    ###########################################################################
    ##################### K means for variable selection ######################
    ###########################################################################
    
    ### The next code is executed if it is not used for independent kmeans (False False False)
    if not con_ponderacion=='False':
        ###### Para la proxima iteracion, los pesos
        etiquetas_p, centroids_p = functions.kmeans(
                                                    X_data_ori,
                                                    numiter,
                                                    centroids_p,
                                                    p_dista=p_dista,
                                                    etiquetas=etiquetas_prev_p,
                                                    indivs=this_indi,
                                                    prev_i=prev_indi,
                                                    robust=robust)
    else:
        etiquetas_p = etiquetas.copy()

    ### Guardar esta variable para la proxima iteracion
    etiquetas_prev_p = etiquetas_p.copy()        
    
    ### Obtain variable importances for each cluster (from highest to lowest)
    importancias_cluster = []
    
    ### Iterate for each cluster
    for clu in range(k):
        ### Keep information for each cluster
        datax_i = pd.DataFrame(X_data_ori)
        datay_i = etiquetas_p.copy()

        ### Create a filter to identify those observations that do not belong to
        ### the cluster, then assign a -1 label to those observations. That way,
        ### there is a binary identification that says whether an observation
        ### belongs to that cluster ot not
        distintos_cluster = np.where(datay_i != clu)

        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)

        ### Obtain importances
        relevancias, _ = functions.variables_relevantes_arbol(datax_i, datay_i, 0)

        importancias_cluster.append(relevancias)

    ### Calculate mean of variable importances
    imp_clus_prom = np.mean(importancias_cluster, axis=0)

    ### Append importance of each iteration
    imp_iters.append(imp_clus_prom)

t_fin = (time.time() - t_ini) / 60

### Create outputs folder in case it does not exists
if not os.path.exists('../data/outputs'):
    os.mkdir('../data/outputs')

### Save tables in CSV format containing the results of the dynamic segmentation
df1 = pd.DataFrame(datos_e)
df1.reset_index().to_csv('../data/outputs/datos_e.csv', header=True, index=False)

df2 = pd.DataFrame(datos_pca)
df2.reset_index().to_csv('../data/outputs/datos_pca.csv', header=True, index=False)

df3 = pd.DataFrame(X_data_df)
df3.reset_index().to_csv('../data/outputs/X_data_df.csv', header=True, index=False)

df4 = pd.DataFrame(etiquetas_glo)

### whether "args" is considered or not
if len(sys.argv) == 1:
    df4.reset_index().to_csv('../data/outputs/NuestraSegDyn.csv', header=True, index=False)
    with open('../data/outputs/execution_time.txt', 'a+') as f:
        f.write(f'\n NuestraSegDyn: {t_fin}')

if len(sys.argv) == 4:
    concat_esc = sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3]
    df4.reset_index().to_csv('../data/outputs/NuestroKmeans_' + concat_esc + '.csv', header=True, index=False)
    with open('../data/outputs/execution_time.txt', 'a+') as f:
        f.write(f'\n NuestraSegDyn_{concat_esc}: {t_fin}')

if len(sys.argv) == 5:
    df4.reset_index().to_csv('../data/outputs/NuestroKmeans_Robust.csv', header=True, index=False)
    with open('../data/outputs/execution_time.txt', 'a+') as f:
        f.write(f'\n NuestroKmeans_Robust: {t_fin}')

df5 = pd.DataFrame(imp_periods_var)
df5.reset_index().to_csv('../data/outputs/imp_periods_var.csv', header=True, index=False)

### The following parameters are numbers without a  dataframe structure

dsimple = pd.DataFrame(columns=('year_i', 'k', 'periodos_incluir', 'scaler_es', 'dim_centroids_ite'))

dimension = np.array(centroids_ite).shape
largo = dimension[2] * dimension[1]

dsimple.loc[len(dsimple)] = [year_i, k, periodos_incluir, scaler_es, dimension]
df6 = pd.DataFrame(dsimple)
df6.reset_index().to_csv('../data/outputs/yeari_k_periods_scaler_es_dimcentroi.csv', header=True, index=False)

centros = np.array(centroids_ite).reshape(-1, largo)
df7 = pd.DataFrame(centros)
df7.reset_index().to_csv('../data/outputs/centroids_ite.csv', header=True, index=False)

datos = pd.DataFrame(datos_pca, columns=['component_1', 'component_2'])
datos = pd.concat([datos_e, datos], axis=1, ignore_index=True)

### Save data
datos.to_csv('../data/outputs/datos.csv', header=True, index=False)

### Save standard scaler
import pickle
with open('../data/outputs/s_scaler.pickle', 'wb') as f:
    pickle.dump(scaler_es, f)