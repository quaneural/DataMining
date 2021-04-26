
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
#Bridget Adding Comment
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
  WHERE date = DATE_SUB("2021-03-25", INTERVAL 30 day)
')
#WHERE date = DATE_SUB(CURRENT_DATE(), INTERVAL 7 day)
#?? Daniel the number of rows does not change if I put an interval of  7 or 30 days
cases_orig = cases

#TOO BIG TO DOWNLOAD??
mobility <- dbGetQuery(con,"
 SELECT * 
  FROM `bigquery-public-data.covid19_google_mobility.mobility_report` 
  WHERE country_region LIKE 'United States'
")


govt_response <- dbGetQuery(con,"
 SELECT * 
  FROM `bigquery-public-data.covid19_govt_response.oxford_policy_tracker` 
  WHERE country_name LIKE 'United_States'
")

#NO COUNTY FIPS CODE or COUNTY NAME??
str(govt_response)
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
#  Begin Feature Importance Analysis  --------------------------------------------------------------------------------------------------
#     CHI SQUARE AND CFS
#  Find most important features across the US   -----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
# My Goal is the get the top 80 of 259 features and feed it into Random Forest for feature importance, then choose top 20
colnames(cases_orig)
death_set = dplyr::select(cases_orig, -county_fips_code, -state_fips_code, -geo_id, -state, -date, -gini_index, -do_date, -county_name, -confirmed_cases)
covid_set = dplyr::select(cases_orig, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -do_date, -county_name, -deaths)
colnames(death_set)
#Remove county_fips_code, geo_id, county_name, confirmed_cases
#Remove county_fips_code, geo_id, county_name, deaths

#Interestingly ChiSquare can be performed without Factoring or Removing N/A
US_death_weights <- death_set %>% chi.squared(deaths ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_death_weights
write.table(US_death_weights, file="chisquare_US_death_weights.csv",sep = ",")

US_covid_weights <- covid_set %>% chi.squared(confirmed_cases ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_covid_weights
write.table(US_covid_weights, file="chisquare_US_covid_weights.csv",sep = ",")


cfs_US_death_features= death_set %>%cfs(deaths ~ ., data = .)
cfs_US_death_features
#[1] "rent_over_50_percent"            "hispanic_pop"                    "other_race_pop"                 
#[4] "income_10000_14999"              "dwellings_2_units"               "housing_built_1939_or_earlier"  
#[7] "male_75_to_79"                   "female_75_to_79"                 "amerindian_including_hispanic"  
#[10] "commute_90_more_mins"            "commuters_by_subway_or_elevated" "male_45_64_high_school"         
#[13] "poverty"                         "high_school_diploma"             "commute_60_more_mins"           
#[16] "hispanic_any_race"    
cfs_US_covid_features=covid_set %>%cfs(confirmed_cases ~ ., data = .)
cfs_US_covid_features
#[1] "father_one_parent_families_with_young_children" "hispanic_male_45_54"                           
#[3] "occupation_sales_office"                        "sales_office_employed

# Find Top 80 features by merging CFS with ChiSquare
#Based on data  March 25, 2021 - April 25, 2021
#Need to rerun if we are to look at earlier dates in 2020
top84USCovidFeatures = dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, confirmed_cases,
father_one_parent_families_with_young_children,hispanic_male_45_54,occupation_sales_office,sales_office_employed,
children,female_under_5,families_with_young_children,male_under_5,male_pop,
in_school,population_3_years_over,female_30_to_34,total_pop,population_1_year_and_over,
in_grades_1_to_4,pop_25_64,in_grades_5_to_8,male_30_to_34,pop_16_over,
commuters_by_car_truck_van,pop_determined_poverty_status,female_pop,in_grades_9_to_12,
female_10_to_14,commuters_16_over,male_25_to_29,civilian_labor_force,male_10_to_14,
pop_in_labor_force,female_25_to_29,commuters_drove_alone,female_5_to_9,pop_25_years_over,
workers_16_and_over,male_5_to_9,employed_pop,households,occupied_housing_units,male_15_to_17,
female_15_to_17,family_households,male_35_to_39,female_35_to_39,male_40_to_44,
male_45_to_49,occupation_sales_office,sales_office_employed,female_45_to_49,male_50_to_54,
female_40_to_44,female_22_to_24,not_hispanic_pop,male_45_to_64,two_cars,
female_50_to_54,married_households,employed_education_health_social,occupation_services,
male_22_to_24,housing_units_renter_occupied,one_year_more_college,owner_occupied_housing_units,
two_parent_families_with_young_children,some_college_and_associates_degree,one_car,
commute_15_19_mins,employed_retail_trade,high_school_including_ged,income_50000_59999,
male_55_to_59,commute_10_14_mins,income_60000_74999,three_cars,female_55_to_59,
not_in_labor_force,management_business_sci_arts_employed,occupation_management_arts,occupation_production_transportation_material,
nonfamily_households,housing_units,white_including_hispanic,high_school_diploma,income_75000_99999,commuters_by_carpool)

top96USDeathFeatures = dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, deaths,
rent_over_50_percent,hispanic_pop,other_race_pop,income_10000_14999,dwellings_2_units,housing_built_1939_or_earlier,male_75_to_79,
female_75_to_79,amerindian_including_hispanic,commute_90_more_mins,commuters_by_subway_or_elevated,male_45_64_high_school,poverty,
high_school_diploma,commute_60_more_mins,hispanic_any_race,children_in_single_female_hh,high_school_including_ged,one_car,
not_in_labor_force,female_pop,children,population_3_years_over, population_1_year_and_over, total_pop,
high_school_diploma,one_parent_families_with_young_children,families_with_young_children, pop_25_64,pop_25_years_over,
in_grades_9_to_12,pop_16_over,female_under_5, male_pop,female_25_to_29,male_under_5,female_5_to_9,in_grades_1_to_4,
pop_determined_poverty_status,less_than_high_school_graduate,male_15_to_17,income_20000_24999,female_80_to_84,family_households,
female_75_to_79,female_70_to_74, in_grades_5_to_8,female_15_to_17,households, occupied_housing_units,male_10_to_14,
female_45_to_49,poverty, in_school, female_40_to_44,  income_25000_29999,income_30000_34999,income_15000_19999,
female_10_to_14, female_50_to_54,male_5_to_9,income_10000_14999, male_45_to_49, male_45_to_64, commuters_by_car_truck_van,
owner_occupied_housing_units,male_25_to_29, female_35_to_39, male_50_to_54, commuters_drove_alone,two_cars,
occupation_production_transportation_material, commuters_16_over,income_35000_39999,pop_in_labor_force,male_40_to_44,
housing_units,housing_units_renter_occupied,male_45_64_high_school,female_22_to_24,nonfamily_households,
households_public_asst_or_food_stamps, not_hispanic_pop,female_55_to_59,male_30_to_34,workers_16_and_over,male_35_to_39,
male_55_to_59,income_40000_44999,one_year_more_college,income_less_10000,civilian_labor_force,occupation_services,
income_50000_59999,employed_pop)

library("relaimpo")
library("randomForest")
#  Random Forest feature importance to sort feature importance
rf <-randomForest(deaths ~ . , data=top96USDeathFeatures , importance=TRUE,ntree=1000)
#Evaluate variable importance
imp = importance(rf, type=1)
imp <- data.frame(predictors=rownames(imp),imp)
# Order the predictor levels by importance
sorted = imp
sorted= sorted[order(sorted$X.IncMSE, decreasing = TRUE),] 
sorted
write.csv(sorted, file = "sortedFeatureImportanceRandomForest-Deaths.csv", row.names = TRUE)

rf <-randomForest(confirmed_cases ~ . , data=top84USCovidFeatures , importance=TRUE,ntree=1000)
imp = importance(rf, type=1)
imp <- data.frame(predictors=rownames(imp),imp)
sorted = imp
sorted= sorted[order(sorted$X.IncMSE, decreasing = TRUE),] 
sorted
write.csv(sorted, file = "sortedFeatureImportanceRandomForest-Covid.csv", row.names = TRUE)


top20DeathFeatures = dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, deaths,
high_school_diploma,high_school_including_ged,male_45_64_high_school,poverty,less_than_high_school_graduate,income_10000_14999,
children_in_single_female_hh,households_public_asst_or_food_stamps,commute_60_more_mins,female_75_to_79,not_in_labor_force,
female_80_to_84,not_hispanic_pop,income_15000_19999,dwellings_2_units,other_race_pop,commute_90_more_mins,
one_parent_families_with_young_children,rent_over_50_percent)

top21CovidFeatures = dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, confirmed_cases,
 father_one_parent_families_with_young_children,in_grades_1_to_4,children, male_5_to_9, female_under_5, families_with_young_children,
male_under_5, hispanic_male_45_54,high_school_including_ged, high_school_diploma,female_15_to_17, in_grades_5_to_8,female_5_to_9,
in_grades_9_to_12, occupation_sales_office, male_10_to_14, female_10_to_14,sales_office_employed, in_school, employed_retail_trade)

#Can we merge these top 20 down to 15?

#Now look at deaths as a percentage of population or deaths per 1000
death_set2 = death_set
covid_set2 = covid_set

death_set2$deathsP1000 = death_set2$deaths*1000/(death_set2$total_pop)
covid_set2$casesP1000= covid_set2$confirmed_cases*1000/(covid_set2$total_pop)

death_set2 = dplyr::select(death_set2, -deaths)
US_death1000_weights <- death_set2 %>% chi.squared(deathsP1000 ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_death1000_weights
write.table(US_death1000_weights, file="chisquare_US_death1000_weights.csv",sep = ",")

covid_set2 = dplyr::select(covid_set2, -confirmed_cases)
US_covid1000_weights <- covid_set2 %>% chi.squared(casesP1000 ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_covid1000_weights
write.table(US_covid1000_weights, file="chisquare_US_covid1000_weights.csv",sep = ",")

cfs_US_death1000_features= death_set2 %>%cfs(deathsP1000 ~ ., data = .)
cfs_US_death1000_features
cfs_US_covid1000_features=covid_set2 %>%cfs(casesP1000 ~ ., data = .)
cfs_US_covid1000_features

top82USDeath1000Features =dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, deaths,total_pop,
amerindian_pop,commuters_by_subway_or_elevated,owner_occupied_housing_units_lower_value_quartile,
owner_occupied_housing_units_upper_value_quartile,owner_occupied_housing_units_median_value,walked_to_work,
dwellings_10_to_19_units,median_rent,vacant_housing_units_for_sale,income_15000_19999,
worked_at_home,female_female_households,renter_occupied_housing_units_paying_cash_median_gross_rent,
father_one_parent_families_with_young_children,father_in_labor_force_one_parent_families_with_young_children,
commute_35_44_mins,commute_35_39_mins,rent_10_to_15_percent,income_100000_124999,white_pop,white_male_55_64,graduate_professional_degree,income_150000_199999,
male_45_64_bachelors_degree,male_45_64_some_college,amerindian_pop,three_cars,
less_one_year_college,mortgaged_housing_units,bachelors_degree_or_higher_25_64,employed_science_management_admin_waste,
bachelors_degree_2,bachelors_degree,some_college_and_associates_degree,masters_degree,commute_25_29_mins,employed_construction,commute_40_44_mins,four_more_cars,
employed_arts_entertainment_recreation_accommodation_food,male_45_64_graduate_degree,income_125000_149999,white_including_hispanic,employed_pop,less_than_high_school_graduate,
income_75000_99999,workers_16_and_over,commuters_by_car_truck_van,male_male_households,
armed_forces,commuters_16_over,income_60000_74999,commute_30_34_mins,management_business_sci_arts_employed,occupation_management_arts,
not_hispanic_pop,commuters_drove_alone,male_45_64_associates_degree,associates_degree,pop_in_labor_force,two_parents_in_labor_force_families_with_young_children,
other_race_pop,civilian_labor_force,employed_wholesale_trade,male_55_to_59,employed_finance_insurance_real_estate,
employed_information,occupation_sales_office,sales_office_employed,different_house_year_ago_different_city,
median_income,employed_other_services_not_public_admin,two_cars,one_year_more_college,million_dollar_housing_units,
dwellings_1_units_detached,income_less_10000,male_60_61,poverty,male_40_to_44,commuters_by_carpool,income_50000_59999,occupation_natural_resources_construction_maintenance)


top82USDeath1000Features$deathsP1000 = top82USDeath1000Features$deaths*1000/(top82USDeath1000Features$total_pop)


top83USCovid1000Features  =dplyr::select(cases_orig, county_fips_code, geo_id, state_fips_code, state, date, county_name, confirmed_cases,total_pop,
amerindian_pop,owner_occupied_housing_units_upper_value_quartile,owner_occupied_housing_units_median_value,owner_occupied_housing_units_lower_value_quartile,
renter_occupied_housing_units_paying_cash_median_gross_rent,median_age,median_rent,percent_income_spent_on_rent,
vacant_housing_units,income_per_capita,employed_agriculture_forestry_fishing_hunting_mining,
commuters_by_subway_or_elevated,commute_less_10_mins,million_dollar_housing_units,median_income,
vacant_housing_units_for_sale,two_parents_in_labor_force_families_with_young_children,commute_30_34_mins,
commute_5_9_mins,commute_45_59_mins,commuters_by_public_transportation,commute_60_more_mins,
walked_to_work,employed_manufacturing,employed_wholesale_trade,different_house_year_ago_same_city,
worked_at_home,commute_90_more_mins,households_retirement_income,group_quarters,commute_35_39_mins,housing_built_1939_or_earlier,
employed_finance_insurance_real_estate,two_parent_families_with_young_children,white_male_45_54,commute_35_44_mins,
black_including_hispanic,masters_degree,white_male_55_64,black_pop,male_65_to_66,male_70_to_74,female_65_to_66,
white_pop,commute_25_29_mins,in_grades_5_to_8,three_cars,female_85_and_over,
commuters_drove_alone,dwellings_1_units_attached,commute_40_44_mins,
commuters_by_bus,black_male_45_54,commute_20_24_mins,male_45_64_graduate_degree,black_male_55_64,aggregate_travel_time_to_work,
employed_education_health_social,employed_transportation_warehousing_utilities,rent_under_10_percent,
male_5_to_9,employed_science_management_admin_waste,graduate_professional_degree,male_22_to_24,male_75_to_79,
male_male_households,occupation_production_transportation_material,female_62_to_64,
less_than_high_school_graduate,commute_60_89_mins,female_70_to_74,
mobile_homes,management_business_sci_arts_employed,occupation_management_arts,male_under_5,mortgaged_housing_units,female_60_to_61,
employed_other_services_not_public_admin,female_80_to_84,income_200000_or_more,income_45000_49999,occupation_natural_resources_construction_maintenance)

top83USCovid1000Features$casesP1000= top83USCovid1000Features$confirmed_cases*1000/(top83USCovid1000Features$total_pop)
#   End Feature Importance Analysis  ----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
# Brainstorming:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Threshold -> Classify by percentages
# Create confusion matrices that classify with a range of thresholds
# Can we:
# - Use Logistic Regression to determine a case/death rate number for counties or states
# - Classifier predicts:
#   - cases (%)
#   - deaths (%)
#   - vaccine distribution (Can this be done without time series analysis?)
# - Assumptions for Feature Selection:
#   - County density should be a major factor
#   - Cases/deaths need to be closely tied to density
#   - Research (maybe just training sample set):
#     -  mask mandate by state 
#     -  vaccines administered by state
# - Businesses implement proof of vaccine policy
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Classification Models:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 1. Logistic Regression
    # 2. Decision Tree
    # 3. Naive Bayes
    # 4. Support Vector Machine
    # 5. Artificial Neural Network

# Create confusion matrices to compare model performance
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





















