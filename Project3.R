# Rex's test one more time!!!!!!!

# Author: Daniel Pompa
# Project 3: Classification

# Prompt:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Classify counties or states in high/low or low/medium/high risk in terms of how affected they would be by a fourth wave.
# These results can be used to prepare the infrastructure and plan possible interventions (e.g., mask mandates, temporarily closing businesses and schools, etc.). 
# Early interventions based on data might dampen a severe outbreak and therefore save lives and shorten the length of necessary closings.

# Prompt Guidelines:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Follow the CRISP-DM framework

# 1. Data Preparation [40 points]
# • Define your classes (e.g., more than x corona-related cases or fatalities per a population of 10000 per week). 
#   Explain why you defined the classes this way (maybe you want to look at the data first).
# • Combine files as needed to prepare the data set for classification. You will need a single table with a class attribute to learn a model.
# • Identify predictive features, create additional features, and deal with missing data (for classification models that cannot handle missing data).

# 2. Modeling [50 points]
# • Prepare the data for training, testing and hyper parameter tuning.
# • Create at least 3 different classification models (different techniques or using different class variables) using the training data. 
#   Discuss each model and the the advantages of each used classification method for your classification task.
# • Assess how well each model performs (use training/test data, cross validation, etc. as appropriate).

# 3. Evaluation [5 points]
# • How useful is your model for your stakeholder? How would you assess the model's value if it was used.

# 4. Deployment [5 points]
# • How would your model be used in practice? What actions would be taken based on your model? How often would the model be updated? Etc.
# Graduate Students: Exceptional Work [10 points]

# Import libraries
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library("tidyverse")
library("nnet")
library("MAP")
library("maps")
library("DT")
library("seriation")
library("FSelector")
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Beginning of Getting Started Code
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##### 

# Calculate rates (per 1000 people) and select important variables. You need more variables.
cases <- cases %>% mutate_if(is.character, factor)
dim(cases)

cases <- cases %>% filter(confirmed_cases > 0) 

# Need to select more variables! Selecting more variables from previous work.
cases <- cases %>% arrange(desc(confirmed_cases)) #%>% select(county_name, state, confirmed_cases, deaths, total_pop, median_income, median_age)
cases <- cases %>% mutate(
  cases_per_10000 = confirmed_cases/total_pop*10000,
  deaths_per_10000 = deaths/total_pop*10000,
  death_per_case = deaths/confirmed_cases)

cases

# dput(colnames(cases))

cases_sel <- cases %>% select(county_name, state, total_pop,
                              nonfamily_households, median_year_structure_built,        
                              female_pop, median_age, white_pop, 
                              black_pop, asian_pop, hispanic_pop, amerindian_pop,
                              commuters_by_public_transportation, 
                              households, median_income, housing_units, 
                              vacant_housing_units, 
                              percent_income_spent_on_rent,
                              employed_pop, unemployed_pop, 
                              in_school, in_undergrad_college,
                              cases_per_10000, deaths_per_10000, death_per_case)

# normalize by population 
cases_sel <- cases_sel %>% mutate(
  nonfamily_households = nonfamily_households / total_pop, 
  female_pop = female_pop / total_pop,
  white_pop = white_pop / total_pop, 
  black_pop = black_pop / total_pop, 
  asian_pop = asian_pop / total_pop, 
  hispanic_pop = hispanic_pop / total_pop, 
  amerindian_pop = amerindian_pop / total_pop,
  commuters_by_public_transportation = commuters_by_public_transportation/ total_pop, 
  households = households / total_pop, 
  housing_units = housing_units / total_pop, 
  vacant_housing_units = vacant_housing_units / total_pop, 
  employed_pop = employed_pop / total_pop, 
  unemployed_pop = unemployed_pop / total_pop, 
  in_school = in_school / total_pop, 
  in_undergrad_college = in_undergrad_college / total_pop 
)

cases_sel
summary(cases_sel)

# Check for missing values and if the data looks fine.
table(complete.cases(cases_sel))

# Check that class variable is a factor! Otherwise, many models will perform regression.
str(cases_sel)

# Check correlation for numeric variables
cm <- cor(cases_sel %>% select_if(is.numeric) %>% na.omit)
hmap(cm, margins = c(14,14))

# Idea: Focus on states with Covid-19 outbreaks
# Use a few states with many cases (training data) to learn a model of how demographics and 
# socioeconomic factors affect fatalities and then apply the model to the other states (test data). 
# I define the class variable by discretizing deaths per a population of 10,000 into below and above 10 deaths.

# Create class variable
# “Bad” means a high fatality rate.
cases_sel <- cases_sel %>% mutate(bad = as.factor(deaths_per_10000 > 10))

# Check if the class variable is very imbalanced.
cases_sel %>% pull(bad) %>% table()

cases_sel %>% group_by(state) %>% 
  summarize(bad_pct = sum(bad == TRUE)/n()) %>%
  arrange(desc(bad_pct))

# Split into training and test data
# I use TX, CA, FL, and NY to train.
cases_train <- cases_sel %>% filter(state %in% c("TX", "CA", "FL", "NY"))
cases_train %>% pull(bad) %>% table()

cases_test <-  cases_sel %>% filter(!(state %in% c("TX", "CA", "FL", "NY")))
cases_test %>% pull(bad) %>% table()

# Select Features
# Plot a map for test data
# See: https://eriqande.github.io/rep-res-web/lectures/making-maps-with-R.html

counties <- as_tibble(map_data("county"))
counties <- counties %>% 
  rename(c(county = subregion, state = region)) %>%
  mutate(state = state.abb[match(state, tolower(state.name))]) %>%
  select(state, county, long, lat, group)
counties  

# Add variables to map data
counties_all <- counties %>% left_join(cases_train %>%  mutate(county = county_name %>% str_to_lower() %>% 
                                                  str_replace('\\s+county\\s*$', '')))

## Joining, by = c("state", "county")
ggplot(counties_all, aes(long, lat)) + 
  geom_polygon(aes(group = group, fill = bad), color = "black", size = 0.1) + 
  coord_quickmap() + scale_fill_manual(values = c('TRUE' = 'red', 'FALSE' = 'grey'))

cases_train %>%  chi.squared(bad ~ ., data = .) %>% 
  arrange(desc(attr_importance)) %>% head()

# We have to remove the variable that was used to create the class variable.
cases_train <- cases_train %>% select(-c(deaths_per_10000))
cases_train %>%  chi.squared(bad ~ ., data = .) %>% 
  arrange(desc(attr_importance)) %>% head()

# Remove more COVID19-related variables.
cases_train <- cases_train %>% select(-death_per_case, -cases_per_10000)

cases_train %>%  chi.squared(bad ~ ., data = .) %>% 
  arrange(desc(attr_importance)) %>% head(n = 10)

# Build a model
fit <- cases_train %>%
  train(bad ~ . - county_name - state,
        data = . ,
        #method = "rpart",
        method = "rf",
        #method = "nb",
        trControl = trainControl(method = "cv", number = 10))

fit
#library(rpart.plot)
#rpart.plot(fit$finalModel, extra = 2)

varImp(fit)
# Note: You should probably take cases per day data instead of the total cases.

# Use the model from hard hit states for the rest of the US (test data)
# caret does not make prediction with missing data
cases_test <- cases_test %>% na.omit
cases_test$bad_predicted <- predict(fit, cases_test)

# Visualize the results
counties_test <- counties %>% left_join(cases_test %>% 
                                          mutate(county = county_name %>% str_to_lower() %>% 
                                                   str_replace('\\s+county\\s*$', '')))

## Joining, by = c("state", "county")

# Ground truth
ggplot(counties_test, aes(long, lat)) + 
  geom_polygon(aes(group = group, fill = bad), color = "black", size = 0.1) + 
  coord_quickmap() + 
  scale_fill_manual(values = c('TRUE' = 'red', 'FALSE' = 'grey'))

# Predictions
ggplot(counties_test, aes(long, lat)) + 
  geom_polygon(aes(group = group, fill = bad_predicted), color = "black", size = 0.1) + 
  coord_quickmap() + 
  scale_fill_manual(values = c('TRUE' = 'red', 'FALSE' = 'grey'))


# Confusion matrix
confusionMatrix(data = cases_test$bad_predicted, ref = cases_test$bad)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End of Getting Started Code
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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





















