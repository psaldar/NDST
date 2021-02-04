# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:46:44 2020

@author: nicol
"""

##### Comparison of clustering techniques ################################


### Imports
import time
import pandas as pd
import functions
import numpy as np
import subprocess
semilla = 10
np.random.seed(semilla)

####### For gapminder experiments
de_gapminder = False ### Activate if its any gapminder experiment
WD_gapminder = False ### Activate only if it is gapminder experiment 3

#### Read data (execute main to generate these datasets)
data_e = pd.read_csv('../data/outputs/data_e.csv')
data_pca = pd.read_csv('../data/outputs/data_pca.csv')
etiquetas_glo_old = pd.read_csv('../data/outputs/SDVI.csv')
X_data_df = pd.read_csv('../data/outputs/X_data_df.csv')
centroids_ite = pd.read_csv('../data/outputs/centroids_ite.csv')

#### Auxiliary variables
year_i = min(data_e['Date'])  
periodos_incluir = max(data_e['Date']) - year_i

### Number of clusters, number of iterations and distance used
k = 3
numiter = 25
p_dista = 2 

if WD_gapminder:
    WD_real = pd.read_csv('../data/real_labels_WD.csv').rename(columns={'Country': 'country'})
    WD_real = WD_real.melt(id_vars=['country'], var_name='Date')
    WD_real['Date'] = WD_real['Date'].astype(int)
    WD_real['value'] = WD_real['value'] - 1




######################### Compared techniques #############################################

################# FKMP1 ####
print('FKMP1')

datos_df_per1 = data_e[data_e['Date'] == year_i]
datos_per1 = np.array(datos_df_per1[datos_df_per1.columns[3:]])

numdata = len(datos_per1)

### FKMP1 is only included in gapminder experiments
if de_gapminder:
    t_ini = time.time()
    centroids = functions.init_centroids(datos_per1, k)
    centroids_p = centroids.copy()
    this_indi = pd.unique(datos_df_per1.country)

    ### kmeans on period 1
    etiquetas, centroids = functions.kmeans(datos_per1,
                                            numiter,
                                            centroids,
                                            p_dista=p_dista,
                                            indivs=this_indi,
                                            prev_i=[])

    ### Labels are the same in all the periods
    FKMP1 = etiquetas_glo_old.copy()
    for perio in range(len(FKMP1)):
        FKMP1.loc[perio] = np.append(perio, etiquetas.copy())

    t_fin = (time.time() - t_ini) / 60
    
    ### Append execution time to csv file
    df5 = pd.DataFrame(FKMP1)
    df5.to_csv('../data/outputs/FKMP1.csv', header=True, index=False)
    with open('../data/outputs/execution_time.txt', 'a+') as f:
        f.write(f'\n FKMP1: {t_fin}')


################### SKMP1 #################################
np.random.seed(semilla)

print('SKMP1')
t_ini = time.time()
year_i = min(data_e['Date'])  
filtro = data_e['Date'] == year_i
X_data_df = data_e[filtro].reset_index(drop=True)
this_indi = pd.unique(X_data_df.country)
tot_indiv = pd.unique(data_e.country)
X_data = np.array(X_data_df[X_data_df.columns[3:]])

numdata = len(X_data)
centroids = functions.init_centroids(X_data, k)

centroids_p = centroids.copy()
###  kmeans
etiquetas, centroids = functions.kmeans(X_data,
                                        numiter,
                                        centroids,
                                        p_dista=p_dista,
                                        indivs=this_indi,
                                        prev_i=[])

### For the rest of the periods, assign each datapoint to the
### cluster of its nearest centroid
etiquetas_km_p1 = []
for periodos in range(periodos_incluir + 1):

    prev_indi = this_indi.copy()
    X_data_df = data_e[data_e['Date'] == year_i + periodos].reset_index(drop=True)
    this_indi = pd.unique(X_data_df.country)
    X_data = np.array(X_data_df[X_data_df.columns[3:]])

    estas_etiqs = []

    for v in range(len(X_data)):
        vec = X_data[v]

        ## Assign to nearest centroid
        mascerc = 0
        valuecerc = np.Inf
        for cent in range(len(centroids)):
            centr = centroids[cent]

            dist = functions.distancia(centr, vec, p_dista)
            if dist < valuecerc:
                mascerc = cent
                valuecerc = dist

        estas_etiqs.append(mascerc)

    etiquetas_km_p1.append(estas_etiqs)


### Labels
SKMP1 = etiquetas_glo_old.copy()
for perio in range(len(SKMP1)):
    SKMP1.loc[perio] = np.append(perio, etiquetas_km_p1[perio].copy())

t_fin = (time.time() - t_ini) / 60
### Append execution time to csv file
df6 = pd.DataFrame(SKMP1)
df6.to_csv('../data/outputs/SKMP1.csv', header=True, index=False)
with open('../data/outputs/execution_time.txt', 'a+') as f:
    f.write(f'\n SKMP1: {t_fin}')


########### TSKM #####
np.random.seed(semilla)
print('TSKM')

### TSKM is only used for gapminder experiments
if de_gapminder or WD_gapminder:
    t_ini = time.time()
    ### Kmeans for time series
    etiquetas_timeseries_kmeans = []
    for periodos in range(periodos_incluir + 1):

        prev_indi = this_indi.copy()
        X_data_df = data_e[data_e['Date'] <= year_i + periodos].reset_index(drop=True)
        this_indi = pd.unique(X_data_df.country)
        X_data = np.array(X_data_df[X_data_df.columns[3:]])

        ##### Each observation is a time series
        numnewvars = X_data.shape[1] * (periodos + 1)
        newxdata = np.zeros((numdata, numnewvars))

        for i in range(numdata):
            auxve = X_data[i * (periodos + 1):(i + 1) * (periodos + 1)][:].reshape(-1)
            newxdata[i] = auxve.copy()

        centrous = functions.init_centroids(newxdata, k)

        ###  kmeans
        etiquetas, centroids = functions.kmeans(newxdata,
                                                numiter,
                                                centrous,
                                                p_dista=p_dista,
                                                indivs=newxdata,
                                                prev_i=[])

        etiquetas_timeseries_kmeans.append(list(etiquetas))

    ### Labels for this year
    TSKM = etiquetas_glo_old.copy()
    for perio in range(len(etiquetas_timeseries_kmeans)):
        TSKM.loc[perio] = np.append(perio, etiquetas_timeseries_kmeans[perio].copy())

    t_fin = (time.time() - t_ini) / 60
    ### Append execution time to csv file
    df7 = pd.DataFrame(TSKM)
    df7.to_csv('../data/outputs/TSKM.csv', header=True, index=False)

    with open('../data/outputs/execution_time.txt', 'a+') as f:
        f.write(f'\n TSKM: {t_fin}')



############# CKM ##############

print('CKM')
t_ini = time.time()
etiquetas_fcm2 = []
for periodos in range(periodos_incluir + 1):

    prev_indi = this_indi.copy()
    X_data_df = data_e[data_e['Date'] <= year_i + periodos].reset_index(drop=True)
    this_indi = pd.unique(X_data_df.country)
    X_data = np.array(X_data_df[X_data_df.columns[3:]])

    centroids = functions.init_centroids(X_data, k)

    if periodos == 0:
        centroids = centroids_p.copy()

    ### kmeans
    etiquetas, centroids = functions.kmeans(X_data,
                                            numiter,
                                            centroids,
                                            p_dista=p_dista,
                                            indivs=X_data,
                                            prev_i=[])

    cluster_membership = etiquetas.copy()
    etiquetas = []

    ### Only look at the labels of current period
    X_df_aux = X_data_df.copy()
    X_df_aux['etiqes'] = list(cluster_membership)
    X_df_aux = X_df_aux[X_df_aux['Date'] == year_i + periodos].reset_index(drop=True)
    cluster_membership_f = list(X_df_aux['etiqes'].values)

    final_clust_mem = cluster_membership_f.copy()

    etiquetas_fcm2.append(final_clust_mem)

### Labels
etiquetas_fuzzycm2 = etiquetas_glo_old.copy()
for perio in range(len(SKMP1)):
    etiquetas_fuzzycm2.loc[perio] = np.append(perio, etiquetas_fcm2[perio].copy())

t_fin = (time.time() - t_ini) / 60

### Append execution time to csv file
df8 = pd.DataFrame(etiquetas_fuzzycm2)
df8.to_csv('../data/outputs/CKM.csv', header=True, index=False)
with open('../data/outputs/execution_time.txt', 'a+') as f:
    f.write(f'\n CKM: {t_fin}')
    
    
################################ Import metrics ########################
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy

######################### List of techniques to compare
escenarios_c = []

### SDVI
escenarios_c.append('SDVI')

### FKMP1
if de_gapminder:
    escenarios_c.append('FKMP1')

### SKMP1
escenarios_c.append('SKMP1')

### TSKM
if de_gapminder:
    escenarios_c.append('TSKM')

### CKM
escenarios_c.append('CKM')

### IKM
escenarios_c.append('IKM')

### RSDVI
escenarios_c.append('RSDVI')

############ Generate labels for SDVI, IKM and RSDVI
for esce in escenarios_c:
    print(esce)
    ### SDVI
    if 'SDVI' in esce:
        subprocess.call("python main.py", shell=False)

    ### IKM
    if 'IKM' in esce:
        argums = 'False False False'
        subprocess.call("python main.py " + argums, shell=False)

    ### RSDVI
    if 'RSDVI' in esce:
        argums = 'True True True True'
        subprocess.call("python main.py " + argums, shell=False)

if not de_gapminder:
    ### These are the labels of the simulated dataset
    data_sim = pd.read_csv('../data/simulated_data.csv')

##################### Performance measures for each scenario
voyen = 0
df_metricas = [] 
metric_names = ['frobenius', 'euclid_diag', 'accuracy', 'precision', 'recall', 'f1score']

for esce in escenarios_c:
    etiquetas_glo = pd.read_csv('../data/outputs/' + esce + '.csv')

    etiquetas_correg = etiquetas_glo.copy()
    new_eti_glo = []
    for i in range(len(etiquetas_glo)):
        grab = np.array(etiquetas_glo.loc[i][1:])
        new_eti_glo.append(grab)
    etiquetas_glo = new_eti_glo.copy()
    etiquetas_fcm2 = []
    etiquetas_reales = []
    for periodos in range(len(etiquetas_glo)):
        cluster_membership_f = etiquetas_glo[periodos].copy()

        ### Labels for period i
        X_data_df = data_e[data_e['Date'] == year_i + periodos].reset_index(drop=True)
        X_df_aux = X_data_df.copy()
        X_df_aux['etiqes'] = list(cluster_membership_f)
        X_df_aux = X_df_aux[X_df_aux['Date'] == year_i + periodos].reset_index(drop=True)

        #### Real labels depend on whether it is simulated or gapminder dataset
        if de_gapminder:

            if WD_gapminder:

                WD_real_i = WD_real[WD_real['Date'] == year_i + periodos].reset_index(drop=True)
                datos_e_i = data_e[data_e['Date'] == year_i + periodos].reset_index(drop=True)

                reales = datos_e_i.merge(WD_real_i[['country', 'value']], how='left', on='country').value.values
            else:
                df_hdi = pd.read_csv('../data/HDI_filled.csv')
                porc_datos_low = 0.25
                porc_datos_high = 0.25
                porc_datos_medium = 1 - porc_datos_high - porc_datos_low
                
                if k==4:
                    porc_datos_medium_dos = 0.5
    
                reales = []
                
                df_hdi_filt_a = df_hdi[df_hdi['variable']==year_i  + periodos]
                df_hdi_filt = df_hdi_filt_a[df_hdi_filt_a['country'].isin(pd.unique(X_data_df['country'].values))]

                phigh = np.percentile(df_hdi_filt_a['HDI'].values, (1 - porc_datos_high)*100)
                plow = np.percentile(df_hdi_filt_a['HDI'].values, (porc_datos_low)*100)
                if k==4:
                    pmeddos = np.percentile(df_hdi_filt_a['HDI'].values, (porc_datos_medium_dos)*100)
                
                clase_pais = []
                
                ### For experiment 1 of gapminder
                if year_i>=2000:
                    yearactual = year_i+periodos
                    yearbuscar = int((yearactual-2000)/3)*3+2000
                    excelldc = pd.read_excel('../data/LDC_data.xls', sheet_name=str(yearbuscar))
                    ldc_countries = list(excelldc[excelldc['Status']=='LDC']["Countries/Indicators"].values)
                voyenn = 0
                
                for a in df_hdi_filt['HDI'].values:
                    ### Gapminder experiment 1
                    if k==3:
                        if a =='..':
                            a=0
                        if a >= phigh:
                            ### Developed
                            clase_pais.append(0) 
                        else:
                            ### If it appears in LDC assign it, else it is developing
                            voypais = df_hdi_filt['country'].values[voyenn]
                            if voypais in ldc_countries:
                                    clase_pais.append(2) 
                            else:
                                    clase_pais.append(1)  
                            
                        voyenn = voyenn+1
                        
                    
                    if k==4:
                        ### Gapminder experiment 2
                        if a =='..':
                            a=0
                        if a >= phigh:
                            clase_pais.append(0)  ### Very high
                        if a <  phigh and a >= pmeddos:
                            clase_pais.append(1)  ### High
                        if a <  pmeddos and a >= plow:
                            clase_pais.append(2)  ### Medium
                        if a< plow:
                            clase_pais.append(3)  ## Low                    
            
                ## Labels for this year
                reales = np.array(clase_pais.copy() )

        else:
            reales = data_sim[data_e['Date'] == year_i + periodos].group.values

        etiquetas_reales.append(reales)
        
        ### Label correction to try to match reference cluster class with found cluster class
        res = functions.correct_labels(X_df_aux.drop(columns='etiqes'), reales, cluster_membership_f)

        etiquetas_fcm2.append(res.etiqes.values)

    ### Labels
    for perio in range(len(etiquetas_glo)):
        etiquetas_correg.loc[perio] = np.append(perio, etiquetas_fcm2[perio].copy())
    etiquetas_glo = pd.DataFrame(etiquetas_correg)

    #### Update saved labels after label correction
    etiquetas_glo.to_csv('../data/outputs/' + esce + '.csv', header=True, index=False)

    new_eti_glo = []
    for i in range(len(etiquetas_glo)):
        grab = np.array(etiquetas_glo.loc[i][1:])
        new_eti_glo.append(grab)
    etiquetas_glo = new_eti_glo.copy()

    ### Calculate performance measures
    matrixnorm = []
    euclid_diag = []
    precisions = []
    recalls = []
    accuracys = []
    f1scores = []

    las_metricas = []
    y_acum_true = []
    y_acum_pred = []

    for co in range(len(etiquetas_glo)):
        y_true = etiquetas_reales[co].copy()
        y_pred = etiquetas_glo[co].copy()

        ### Remove outliers for the measures
        y_pred = y_pred[y_true >= 0]
        y_true = y_true[y_true >= 0]

        cla = classification_report(y_true, y_pred, output_dict=True)
        precisions.append(cla['macro avg']['precision'])
        recalls.append(cla['macro avg']['recall'])
        f1scores.append(cla['macro avg']['f1-score'])
        accuracys.append(accuracy_score(y_true, y_pred))
        confu_pred = confusion_matrix(y_true, y_pred)
        confu_true = confusion_matrix(y_true, y_true)
        matrixnorm.append(scipy.linalg.norm(confu_true - confu_pred, ord='fro'))
        euclid_diag.append(scipy.linalg.norm(np.diag(confu_true) - np.diag(confu_pred), ord=2))

        ### For the global confusion matrix
        y_acum_true.extend(y_true.copy())
        y_acum_pred.extend(y_pred.copy())

    ### Metrics for global confusion matrix
    metrics_acum = []
    cla_ac = classification_report(y_acum_true, y_acum_pred, output_dict=True)

    confu_pred_ac = confusion_matrix(y_acum_true, y_acum_pred)
    confu_true_ac = confusion_matrix(y_acum_true, y_acum_true)
    metrics_acum.append(scipy.linalg.norm(confu_true_ac - confu_pred_ac, ord='fro'))  ## frobenius norm
    metrics_acum.append(scipy.linalg.norm(np.diag(confu_true_ac) - np.diag(confu_pred_ac), ord=2))  ## diagonal euclidean distance
    metrics_acum.append(accuracy_score(y_acum_true, y_acum_pred))  ## accuracy
    metrics_acum.append(cla_ac['macro avg']['precision'])  ## precision
    metrics_acum.append(cla_ac['macro avg']['recall'])  ## recall
    metrics_acum.append(cla_ac['macro avg']['f1-score'])  ## f1 score

    if voyen == 0:
        df_acum_m = pd.DataFrame(pd.Series(metrics_acum, name=esce))
        df_acum_m.index = metric_names
    else:
        df_acum_m[esce] = metrics_acum

    las_metricas.append(matrixnorm)
    las_metricas.append(euclid_diag)
    las_metricas.append(accuracys)
    las_metricas.append(precisions)
    las_metricas.append(recalls)
    las_metricas.append(f1scores)


    for metri_n in range(len(las_metricas)):
        metri = las_metricas[metri_n]

        if voyen == 0:
            df_metr = pd.DataFrame(pd.Series(metri, name=esce))
            df_metr.index = pd.unique(data_e['Date'].values)
            df_metricas.append(df_metr.copy())
        else:
            df_metr = df_metricas[metri_n]
            df_metr[esce] = metri
            df_metricas[metri_n] = df_metr.copy()

    voyen = voyen + 1

### Save metrics in an Excel file
path_metr = '../data/outputs/metrics.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(path_metr, engine='xlsxwriter')

### Create global confusion matrix metrics sheet
df_acum_m.to_excel(writer, sheet_name='Global_metrics')

## Create sheets for metrics for each period
voye = 0
for df_met in df_metricas:
    df_met.to_excel(writer, sheet_name=metric_names[voye])
    voye = voye + 1

writer.save()


### Save real labels
dfre = pd.DataFrame(etiquetas_reales)
dfre.reset_index().to_csv('../data/outputs/real_labels.csv', header=True, index=False)