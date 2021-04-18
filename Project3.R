
# Author: Daniel Pompa
# Project 3: Classification
# Last Updated: 4/17/21

# Prompt:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Classify counties or states in high/low or low/medium/high risk in terms of how affected they would be by a fourth wave.
# These results can be used to prepare the infrastructure and plan possible interventions (e.g., mask mandates, temporarily closing businesses and schools, etc.). 
# Early interventions based on data might dampen a severe outbreak and therefore save lives and shorten the length of necessary closings.

# Import libraries
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library("tidyverse")
library("nnet")
library("MAP")
library("DT")
library("seriation")
library("caret")
library("dplyr")
library("DBI")

# Establish connection to bigrquery database and query data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con <- dbConnect(
  bigrquery::bigquery(),
  project = "project-1-data-mining",
  dataset = "covid19_last_week"
)

cases <- dbGetQuery(con,'
 SELECT *
  FROM `bigquery-public-data.covid19_usafacts.summary` covid19
  JOIN `bigquery-public-data.census_bureau_acs.county_2017_5yr` acs
  ON covid19.county_fips_code = acs.geo_id
  WHERE date = DATE_SUB(CURRENT_DATE(), INTERVAL 7 day)
')

# Getting Started Code
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##### 

cases <- cases %>% mutate_if(is.character, factor)
dim(cases)

cases <- cases %>% filter(confirmed_cases > 0) 

# Need to select more variables! Selecting more variables from previous work.
cases <- cases %>% arrange(desc(confirmed_cases)) %>% select(county_name, state, confirmed_cases, deaths, total_pop, median_income, median_age)
cases <- cases %>% mutate(
  cases_per_10000 = confirmed_cases/total_pop*10000,
  deaths_per_10000 = deaths/total_pop*10000,
  death_per_case = deaths/confirmed_cases)

cases


#####
# Classification Models:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 1. Logistic Regression
    # 2. Decision Tree
    # 3. Naive Bayes
    # 4. Support Vector Machine
    # 5. Artificial Neural Network

# Create confusion matrices to compare model performance
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





















