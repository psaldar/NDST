# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:20:05 2019

@author: Pablo Saldarriaga
"""
### Packages needed to run this script
import logging
import numpy as np
import pandas as pd
from sklearn.tree import ExtraTreeClassifier
#%%
logger = logging.getLogger(__name__)
#%%
### This function trains a supervised tree based machine learning model
### to obtain the relevance of the variables for each cluster. The attribute
### feature_importance from the trained model is used to  obtain the relevances.
### the imput data of this function correponds to a dataframe X containing the
### independent variables and dataframe Y contains the dependent variable
def variables_relevantes_arbol(X, Y, alpha=None):

    if len(X) == 0:
        logger.info("No information was passed")
        return []

    features = list(X.columns)

    if alpha == None:
        alpha = 1.0 / len(features)
        logger.info('Aceptance threshold for variable importance is calculated: {0}'.format(alpha))

    try:
        model = ExtraTreeClassifier()
        model.fit(X, Y)

        importance = model.feature_importances_

        relevant_features = []
        for i in range(len(features)):
            if importance[i] > alpha:
                relevant_features.append(features[i])

    except Exception as e:
        logger.info('Error with the tree based model, : There was not relevant variables found{0}'.format(e))
        relevant_features = []

    return importance, relevant_features


### this function correspond to a distance function, in which the distance
### may change according to the desired norm (p)
def distancia(a, b, p, robust=False):

    if robust:
        dist = euclidea_robusta(a,b)
    else:
        dist = np.linalg.norm(a - b, p)

    return dist

### The next function corresponds to the robust version of the euclidean
### distance proposed in this research
def euclidea_robusta(a,b):
    
    porc_recortar = 0.9
    
    n = len(a)
    diff = (a-b)**2
    dats_conservar = int(n*porc_recortar)
    ind = np.argpartition(diff, dats_conservar)[:dats_conservar]
    
    prom = np.mean(diff[ind])
    
    dist = np.sqrt(n*porc_recortar*prom)
    
    
    return dist

### This function randomly initializes the centroids in the kmeans method for
### k groups
def init_centroids(X_data, k):
    logger.info('Initialization of {0} centroids'.format(k))
    centroids = []
    numdata = len(X_data)
    for i in range(k):
        centroids.append(X_data[np.random.randint(0, numdata)])

    return centroids


### Programmed kmeans method
def kmeans(X_data, numiter, centroids, p_dista=2, etiquetas=[], indivs=None, 
           prev_i=None, robust = False):
    logger.info('Kmeans algorithm starts')

    if len(etiquetas) == 0:
        logger.info('Create labels that were not passed')
        
        ### Current labels for each observation
        etiquetas = np.ones(len(indivs)) * -1  ### At first, any observation has a label

    else:

        ### Add the -1 label to the new individuals
        if not list(indivs) == list(prev_i):
            voyen = 0
            new_et = []
            for i in indivs:
                if i in prev_i:
                    posi_p = list(prev_i).index(i)
                    new_et.append(etiquetas[posi_p])
                    voyen = voyen + 1
                else:
                    new_et.append(-1)

            etiquetas = np.array(new_et.copy())


    ### Start iteratrions
    for it in range(numiter):
        logger.info('Iteration {0} out of {1} for kmeans algorithm'.format(it, numiter))
        
        ### On each iteration, we iterate over each observation
        for element in range(len(X_data)):

            np.seterr(all='raise')
            
            ### Evaluate the distance to each centroid. We add 0.00001 to avoid
            ### divisions by 0 on other processes.
            distc = []
            for c in centroids:
                distc.append(distancia(X_data[element], c, p_dista, robust) + 0.00001)

            ### Find the closest centroid
            nearest_centroid = np.argmin(distc)

            ### Assign the observation to the found cluster
            etiquetas[element] = nearest_centroid

            ### RE-calculate the centroid
            if robust:   
                centroids[nearest_centroid] = np.median(X_data[np.where(etiquetas == nearest_centroid)], axis=0)
            else:
                centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas == nearest_centroid)], axis=0)

        centroids = np.array(centroids)

    logger.info('Enf of kmeans')
    return etiquetas, centroids


def correct_labels(data, reales, predichos):
    ### get real centroids
    centroids = []
    for i in range(max(reales)+1):
        filt = reales==i
        data_gi = data[filt].reset_index(drop = True)
        centroids.append([i, np.array(data_gi[data_gi.columns[3:]].mean())])
    
    ### get predicted centroids
    centroids_pred = []
    for i in range(int(max(predichos))+1):
        filt = predichos==i
        data_gi = data[filt].reset_index(drop = True)
        centroids_pred.append([len(data_gi), i, np.array(data_gi[data_gi.columns[3:]].mean())])
    
    centroids_pred.sort(reverse = True)
    
    ### Assign predicted centroids to real centroids
    change_centroids = {}
    for cp in centroids_pred:
        distances = []
        for c in centroids:
            distances.append(np.linalg.norm(c[1] - cp[2], 2))
            
        ### get closest centroid
        real_label = np.argmin(distances)
        change_centroids[cp[1]]=real_label
        centroids[real_label][1] = np.ones(centroids[real_label][1].shape )*np.inf
    
    ### correct labels
    res = []
    for key in change_centroids.keys():
        data_gi = data[predichos == key].reset_index(drop = True)
        data_gi['etiqes'] = change_centroids[key]
        res.append(data_gi)
        
    df_final = pd.concat(res).sort_values(by = 'index').reset_index(drop = True)
    
    return df_final