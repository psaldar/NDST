# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:49:42 2020

@author: Pablo Saldarriaga
"""
import glob
import pandas as pd
import random
random.seed(10)

### Define path in which gapminder files are stored, this path must end with *.csv
### since it indicates all csv files from that path are going to be processed
PATH = r'..\data\raw_gapminder\*.csv'

### Scenario 1, considers information between 2000 and 2015. k = 3
### Scenario 2, considers information between 1980 and 2015, k = 4
### Scenario 3, considers information between 2000 and 2016, k = 4

### Set experiment to generate the dataset
experiment = 3

if experiment == 1:
    ### define range of years to be included in the scenario
    min_year = 2000
    max_year = 2015

if experiment == 2:
    ### define range of years to be included in the scenario
    min_year = 1980
    max_year = 2015

if experiment ==3:
    PATH=r'..\data\raw_gapminder_world_development\*.csv'
    ### define range of years to be included in the scenario
    min_year = 2000
    max_year = 2016
    
year_range = list(range(min_year, max_year + 1))

### Read gapminder information and process it in a single dataframe
data = pd.DataFrame()
for item in glob.glob(PATH):
    temp = pd.read_csv(item)
    temp = temp.melt(id_vars='country', var_name="Date", value_name=item.split("\\").pop().replace('.csv', ''))
    if data.empty:
        data = temp
    else:
        data = pd.merge(data, temp, how='outer', on=['country', 'Date']).reset_index(drop=True)

### Sort the dataframe and keep information from years under consideration
data = data.sort_values(['country', 'Date'])
data['Date'] = data['Date'].astype(int)
data = data[data['Date'].isin(year_range)].reset_index(drop=True)

### Keep variables for scenario 1
if experiment == 1:
    data = data[[
        'Date', 'country', 'children_and_elderly_per_100_adults', 'children_per_woman_total_fertility',
        'child_mortality_0_5_year_olds_dying_per_1000_born', 'total_gdp_us_inflation_adjusted',
        'gdp_per_capita_yearly_growth', 'income_per_person_gdppercapita_ppp_inflation_adjusted',
        'life_expectancy_years', 'mean_years_in_school_men_25_to_34_years', 'mean_years_in_school_women_25_to_34_years',
        'internet_users'
    ]]

### Keep variables for scenario 2
if experiment == 2:
    data = data[[
        'Date', 'country', 'children_and_elderly_per_100_adults', 'children_per_woman_total_fertility',
        'child_mortality_0_5_year_olds_dying_per_1000_born', 'gdp_per_capita_yearly_growth',
        'income_per_person_gdppercapita_ppp_inflation_adjusted', 'life_expectancy_years',
        'mean_years_in_school_men_25_to_34_years', 'mean_years_in_school_women_25_to_34_years',
        'population_density_per_square_km', 'population_growth_annual_percent', 'population_total'
    ]]

### We are going to select the variables to be included, considering missing
### entries in the data, so we keep a rational number of countries and variables
### to work with
data_aux = data.copy()

### Calculate for each year, for each variable, how many countries report 
### information
for col in data_aux.columns:
    if not col in ['country', 'Date']:
        data_aux[col] = (~data_aux[col].isna()).astype(int)

info_data = data_aux.groupby('Date').sum().reset_index()

### Obtain the count of countries in the dataset
cant_paises = len(data['country'].unique())

### we are going to keep those variables in which for every year, have at least
### information for "cutoff_countries" countries.

### Information for scenario 3
if 'world_development' in PATH and experiment == 3:
    list_paises = list(pd.read_csv('../data/real_labels_WD.csv')['Country'].values)
    data = data[data.country.isin(list_paises)].reset_index(drop=True)
    
    ### countries cutoff for scenario 3
    cutoff_countries = 140
else:
    ### countries cutoff for scenarios 1 and 2
    cutoff_countries = 180

### Get the variables to be kept in the dataset
summary = (info_data >= cutoff_countries).sum().reset_index()
vars_to_keep = summary[summary[0] == len(year_range)]['index']

### Once the variables are defined, the next step correspond to obtain the
### countries to include in the dataset to be analyzed. Thus, for each country
### for each variable, we found the number of years that the country reports
### information

data_country = data.copy()

for col in data_country.columns:
    if not col in ['country', 'Date']:
        data_country[col] = (~data_country[col].isna()).astype(int)

all_cols = ['country', *vars_to_keep]
info_data_country = data_country[all_cols].groupby('country').sum()

### We are going to keep the countries that report information for all years
### under study
summary_country = (info_data_country >= len(year_range)).sum(axis=1).reset_index()
countries_to_keep = summary_country[summary_country[0] == len(vars_to_keep)]['country']

### Finally, create a dataframe that contains only the variables to be kept
### as well as the countries to be analyzed
data_final = data[data.country.isin(countries_to_keep.values)][all_cols].reset_index(drop=True)

print(f'There are  {len(data_final.country.unique())} countries in the dataset')
print(data_final.columns)

### save final dataset according to the selected experiment
data_final.to_csv(f'../data/data_gapminder_experiment{experiment}.csv', index=False, sep=',')