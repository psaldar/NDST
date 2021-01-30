A new segmentation approach using dynamic variables on individuals
=======================================================================================


## Description

 This repository contains all data files required to reproduce the analysis in the work "A new segmentation approach using dynamic variables on individuals", whose authors are Nicolás Prieto, Henry Laniado and Juan Carlos Monroy.

#### "A new segmentation approach using dynamic variables on individuals"
============================================================================

Nicolás Prieto<sup>1</sup>, Henry Laniado<sup>2</sup> and Juan Carlos Monroy<sup>3</sup> 

<sup>1</sup> Master of Data Science and Analytics, Universidad EAFIT, Medellin, Colombia

<sup>2</sup> Department of Mathematical Sciences, Universidad EAFIT, Medellin, Colombia

<sup>3</sup> Department of Marketing, Universidad EAFIT, Medellin, Colombia


### Abstract 

The problem of dynamic segmentation consists in finding the segments in which the individuals of a population must be grouped in different periods of time, considering that, as time passes, the general characteristics of the population will be changing, and therefore the segments must be evolving through time. In the literature, diverse techniques have been proposed to achieve the effect of obtaining segments that change through time. However, these techniques have not focused on balancing correctly the importance between past and present information. In this study, a new dynamic segmentation technique is proposed, which observes past behaviors to weight the importance of the variables in a clustering technique and uses the real values of the observations for the current period to find the clusters and to obtain a better segmentation. In addition, an alternative robust version of the proposed dynamic segmentation technique is presented, where some changes are made to have a better performance in presence of case-wise and cell-wise outliers. The performance of both proposals introduced in this work are compared with classical and recent techniques introduced in the literature, in simulated and real datasets with and without outliers. Results show that, for data without outliers, the proposed non-robust dynamic segmentation technique usually outperforms the other dynamic segmentation techniques, but when outliers are included in the dataset, the robust version of the proposed dynamic segmentation technique outperforms both the non-robust version and the other techniques presented in the literature.

## Computational experiments

This repository contains all data files and scripts needed to replicate the results presented in this work. All experiments were programmed and executed on Python 3. In the SDVI.yml file it is possible to find the packages needed to run the code as well as their corresponding versions.

In this repository, two folders are found. First, the "scripts" folder contains all python files that were used to tun the experiments. The functions.py script contains all relevant functions used across all other scripts (functions such as the k-means algorithm, Euclidean distance, robust Euclidean distance, etc.). 

In addition, two principal files that are used to generate the processed datasets correspond to sim_data.py and process_gapminder_data.py. The first script contains the code which simulates a dataset for a dynamic segmentation problem (the simulation process corresponds to the one described in the work), having the possibility to include case-wise or cell-wise outliers. In this script, all parameters described in the work can be modified to simulate the data, and its output corresponds to a csv file in the data folder which is used when comparing the different dynamic segmentation techniques. 

On the other hand, the process_gapminder_data.py scripts processes historical socio-economic and demographic information retrieved from the Gapminder website. It processes the information according to the desired scenario (the three scenarios considered in this script are explained in detail in the work). The only parameter that must be set in this script corresponds to the "experiment" variable at the beginning of the scripts, which can take values from 1, 2 or 3 (referring to the experiment to be processed). The output of this script corresponds to a csv file in the data folder which will be used as an input when executing the script with the dynamic segmentation techniques. 

The next script that must be run after simulating the data or processing the gapminder information is main.py. In this script, the dataset to be read, number of iterations of the k-means algorithm and the number of clusters to consider (k) must be defined, and it applies the proposed dynamic segmentation technique on the dataset. This scripts outputs some csv files in the folder "data/outputs" (which is created in the script), which contain the dataset projected on their first two principal components, the centroids of each group at each period, the variable importances across all periods and the labels assigned by the technique to each observation in every period. 

Finally, the techniques_comparison.py file receives as an input the output files from the main.py script. Also, the number of clusters (k) and number of iterations to be considered must be defined before running this script. Once all parameters are set, this script executes all other techniques considered in this work to compare them with the proposed ones, and it returns an excel file which contains all performance metrics for each of the techniques.
