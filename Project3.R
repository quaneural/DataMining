
# Authors: Daniel Pompa, Bridget Beamon, Rex Lin
# Project 3: Classification
# Write-up on OneDrive: https://1drv.ms/w/s!Agvju_On0zbph3KsAWXVzNrx2uaB?e=tVaMK5


# Prompt:
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Classify counties or states in high/low or low/medium/high risk in terms of how affected they would be by a fourth wave.
# These results can be used to prepare the infrastructure and plan possible interventions (e.g., mask mandates, temporarily closing businesses and schools, etc.). 
# Early interventions based on data might dampen a severe outbreak and therefore save lives and shorten the length of necessary closings.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# Import libraries
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library("tidyverse")
library("nnet")
library("neuralnet")
library("MASS")
library("MAP")
library("maps")
library("DT")
library("seriation")
library("FSelector")
library("caret")
library("dplyr")
library("DBI")
library("relaimpo")
library("randomForest")
library("fuzzyjoin")
library("data.table")
library("rpart")
library("rpart.plot")
library("caTools")
library("caret")
library("e1071")
library("ROCR")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
  WHERE CAST(date AS DATETIME) BETWEEN "2021-04-21" AND "2021-04-27"
')

mobility <- dbGetQuery(con,'
  SELECT * 
  FROM `bigquery-public-data.covid19_google_mobility.mobility_report` 
  WHERE country_region LIKE "United States" AND
  CAST(date AS DATETIME) BETWEEN "2021-04-21" AND "2021-04-27"
')

govt_response <- dbGetQuery(con,'
 SELECT * 
  FROM `bigquery-public-data.covid19_govt_response.oxford_policy_tracker` 
  WHERE country_name LIKE "United_States"
')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Preprocessing/Data Cleaning for Predicting Covid Cases
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

cases_orig <- cases

# Joining cases_normalized data with mobility query data with inital cleaning of merged dataframe "cases_mob" -> "cases_orig"
mob <- mobility %>% rename(county_fips_code=census_fips_code)

# Number of counties in US is 3006 => 7 day report for US should have 15042 observables
cases_mobility <- regex_inner_join(cases_orig, mob, by='county_fips_code')
cases_mob <- cases_mobility[!is.na(cases_mobility$parks_percent_change_from_baseline), ]
cases_mob <- cases_mob[!is.na(cases_mob$county_fips_code.x), ] 
cases_mob <- cases_mob %>% rename(date=date.x)
cases_mob <- cases_mob %>% rename(county_fips_code=county_fips_code.x)
cases_mob <- setDT(cases_mob)[,(249:270) :=NULL] 

cases_first_day <- cases_mob %>% filter(date == '2021-04-21') 
cases_first_day <- cases_mob[1:2110,]
cases_last_day <- cases_mob %>% filter(date == '2021-04-27') 
cases_one_day <- cases_mob %>% filter(date == '2021-04-27') 

cases_one_day <- cases_one_day %>% mutate(delta = cases_last_day$confirmed_cases - cases_first_day$confirmed_cases)
cases_one_day <- cases_one_day %>% filter(delta > 0)
cases_normalized <- cases_one_day %>% mutate(cases_norm = cases_one_day$delta/total_pop*100000)
summary(cases_normalized$cases_norm)

#Determine thresholds to match CDC
plot(cases_normalized$cases_norm, xlim=c(1,1000), log='x', type="l", col="orange", lwd=5)#, xlab="time", ylab="concentration")

# For Decision Tree and KNN models, here is a quick way to remove mobility features.
cases_normalized <- setDT(cases_normalized)[,(249:252) :=NULL] 
summary(cases_normalized$cases_norm)

# Cases' Classes: low < 5000 <= moderate < 15000 <= high
# Add Covid Labels
cases_normalized["Class"] = "MEDIUM"
cases_normalized$Class[5000 < cases_normalized$cases_norm | cases_normalized$cases_norm <= 15000] = "MEDIUM"
cases_normalized$Class[(cases_normalized$cases_norm <= 5000)] = "LOW"
cases_normalized$Class[(cases_normalized$cases_norm > 15000)] = "HIGH"
str(cases_normalized)
table(cases_normalized$Class)

# Train/Test Split Cases set
set.seed(100)
spl = sample.split(cases_normalized$Class, SplitRatio = 0.70)
CovidDataTrain = subset(cases_normalized, spl == TRUE)
CovidDataTest = subset(cases_normalized, spl == FALSE)
table(CovidDataTrain$Class)
table(CovidDataTest$Class)

#REMOVE confirmed cases and other features we do not want to train on
CovidDataTrain2= dplyr::select(CovidDataTrain, -deaths, -delta, -cases_norm, -county_fips_code, -geo_id, -state_fips_code, -state, -date, -county_name, -confirmed_cases, -total_pop)
CovidDataTrainScaled = CovidDataTrain2 %>% dplyr::mutate_if(is.numeric, scale)
CovidDataTest2=dplyr::select(CovidDataTest,  -deaths, -delta, -cases_norm, -county_fips_code, -geo_id, -state_fips_code, -state, -date, -county_name, -confirmed_cases, -total_pop)
CovidDataTestScaled=CovidDataTest2%>% dplyr::mutate_if(is.numeric, scale)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Preprocessing/Data Cleaning for Predicting Covid Deaths
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

deaths_first_day <- cases_mob %>% filter(date == '2021-04-21')
deaths_first_day <- cases_mob[1:2110,]
deaths_last_day <- cases_mob %>% filter(date == '2021-04-27')
deaths_one_day <- cases_mob %>% filter(date == '2021-04-27')

deaths_one_day <- deaths_one_day %>% mutate(delta = deaths_last_day$deaths - deaths_first_day$deaths)
deaths_one_day <- deaths_one_day %>% filter(delta > 0)
deaths_normalized <- deaths_one_day %>% mutate(deaths_norm = deaths_one_day$delta/total_pop*100000)
summary(deaths_normalized$deaths_norm)

#Determine thresholds to match CDC
plot(deaths_normalized$deaths_norm, xlim=c(1,300), log='x', type="l", col="red", lwd=5)#, xlab="time", ylab="concentration")

# For Decision Tree and KNN models, here is a quick way to remove mobility features.
deaths_normalized <- setDT(deaths_normalized)[,(249:252) :=NULL] 
summary(deaths_normalized$deaths_norm)

# Deaths' Classes: low < 100 <= moderate < 300 <= high
# Add Death Labels
deaths_normalized["Class"] = "MEDIUM"
deaths_normalized$Class[100 < deaths_normalized$deaths_norm | deaths_normalized$deaths_norm <= 300] = "MEDIUM"
deaths_normalized$Class[(deaths_normalized$deaths_norm <= 100)] = "LOW"
deaths_normalized$Class[(deaths_normalized$deaths_norm > 300)] = "HIGH"
str(deaths_normalized)
table(deaths_normalized$Class)

# Train/Test Split Death set
set.seed(100)
spl = sample.split(deaths_normalized$Class, SplitRatio = 0.70)
deathDataTrain = subset(deaths_normalized, spl == TRUE)
deathDataTest = subset(deaths_normalized, spl == FALSE)
table(deathDataTrain$Class)
table(deathDataTest$Class)

#REMOVE confirmed cases and other features we do not want to train on
deathDataTrain2= dplyr::select(deathDataTrain,  -deaths, -delta, -deaths_norm, -county_fips_code, -geo_id, -state_fips_code, -state, -date, -county_name, -confirmed_cases, -total_pop)
deathDataTrainScaled = deathDataTrain2 %>% dplyr::mutate_if(is.numeric, scale)
deathDataTest2=dplyr::select(deathDataTest, -deaths, -delta, -deaths_norm, -county_fips_code, -geo_id, -state_fips_code, -state, -date, -county_name, -confirmed_cases, -total_pop)
deathDataTestScaled=deathDataTest2%>% dplyr::mutate_if(is.numeric, scale)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#  Feature Importance on Normalized Data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
covid_set3 = dplyr::select(cases_normalized, -Class, -total_pop, -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths)
colnames(covid_set3)
US_covidNorm_weights <- covid_set3 %>% chi.squared(cases_norm ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_covidNorm_weights
#write.table(US_covidNorm_weights, file="chisquare_US_covidNorm_weights.csv",sep = ",")

cfs_US_covidNorm_features=covid_set3 %>%cfs(cases_norm ~ ., data = .)
cfs_US_covidNorm_features
#[1] "vacant_housing_units_for_sale"           "hispanic_male_55_64"                     "male_45_64_high_school"                 
#[4] "workplaces_percent_change_from_baseline


death_set3 = dplyr::select(deaths_normalized, -Class, -total_pop,  -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths)
colnames(death_set3)
US_deathsNorm_weights <- death_set3 %>% chi.squared(deaths_norm ~ ., data = .) %>%
  as_tibble(rownames = "feature") %>%
  arrange(desc(attr_importance))
US_deathsNorm_weights
#write.table(US_deathsNorm_weights, file="chisquare_US_deathsNorm_weights.csv",sep = ",")

cfs_US_deathsNorm_features=death_set3 %>%cfs(deaths_norm ~ ., data = .)
cfs_US_deathsNorm_features
#[1] "median_age"                              "amerindian_pop"                          "other_race_pop"                         
#[4] "percent_income_spent_on_rent"            "dwellings_2_units"                       "commuters_by_subway_or_elevated"        
#$[7] "workplaces_percent_change_from_baseline
topUSDeathNormFeatures= dplyr::select(deaths_normalized, -Class, -total_pop,  -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths,
                                      median_age,amerindian_pop,other_race_pop,percent_income_spent_on_rent,dwellings_2_units,commuters_by_subway_or_elevated,workplaces_percent_change_from_baseline,
                                      male_45_64_bachelors_degree,male_75_to_79,male_45_64_graduate_degree,walked_to_work,black_pop,different_house_year_ago_same_city,
                                      housing_units,not_in_labor_force,female_25_to_29,income_30000_34999,dwellings_2_units,male_pop,pop_25_years_over,one_car,income_150000_199999,
                                      male_22_to_24,nonfamily_households,poverty,no_cars,white_male_45_54,households,occupied_housing_units,commute_5_9_mins,amerindian_including_hispanic,
                                      commute_less_10_mins,income_50000_59999,male_15_to_17,employed_pop,income_35000_39999,employed_education_health_social,female_5_to_9,
                                      occupation_services,family_households,group_quarters,employed_arts_entertainment_recreation_accommodation_food,owner_occupied_housing_units_upper_value_quartile,
                                      rent_10_to_15_percent,rent_40_to_50_percent,other_race_pop,rent_over_50_percent,housing_units_renter_occupied,no_car,
                                      dwellings_50_or_more_units,dwellings_1_units_attached,female_75_to_79,housing_built_1939_or_earlier,father_one_parent_families_with_young_children,
                                      male_25_to_29,male_18_to_19,rent_25_to_30_percent,male_80_to_84,asian_male_45_54,male_62_64,female_85_and_over,male_50_to_54,
                                      female_20,income_10000_14999,commuters_by_carpool,workers_16_and_over,income_125000_149999,commuters_16_over,income_per_capita,
                                      male_45_64_less_than_9_grade,children_in_single_female_hh,owner_occupied_housing_units,rent_35_to_40_percent,hispanic_male_45_54,
                                      commute_35_39_mins,income_75000_99999,male_67_to_69,employed_agriculture_forestry_fishing_hunting_mining,in_grades_5_to_8,
                                      in_grades_1_to_4,male_45_to_64,income_15000_19999,two_parents_in_labor_force_families_with_young_children,two_parents_father_in_labor_force_families_with_young_children,
                                      mobile_homes,hispanic_pop,hispanic_any_race)


topUSCovidNormFeatures = dplyr::select(cases_normalized, -Class, -total_pop, -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths,
                                       vacant_housing_units_for_sale,hispanic_male_55_64,male_45_64_high_school,workplaces_percent_change_from_baseline,
                                       population_3_years_over,pop_25_64,male_45_64_some_college,commuters_by_car_truck_van,female_20,income_150000_199999,
                                       population_1_year_and_over,female_45_to_49,female_pop,four_more_cars,one_year_more_college,employed_retail_trade,
                                       some_college_and_associates_degree,female_18_to_19,white_male_45_54,graduate_professional_degree,employed_pop,
                                       income_25000_29999,masters_degree,male_10_to_14,dwellings_1_units_attached,male_5_to_9,income_30000_34999,female_15_to_17,
                                       rent_40_to_50_percent,female_5_to_9,associates_degree,male_65_to_66,one_car,commute_less_10_mins,two_or_more_races_pop,
                                       families_with_young_children,male_45_64_grade_9_12,median_income,pop_in_labor_force,unemployed_pop,income_35000_39999,
                                       children_in_single_female_hh,not_us_citizen_pop,male_80_to_84,male_15_to_17,female_10_to_14,housing_built_1939_or_earlier,
                                       male_75_to_79,male_45_to_49,in_grades_9_to_12,housing_units,male_22_to_24,owner_occupied_housing_units_median_value,
                                       commute_25_29_mins,commute_40_44_mins,dwellings_2_units,housing_built_2005_or_later,income_100000_124999,
                                       different_house_year_ago_same_city,pop_16_over,rent_10_to_15_percent,dwellings_10_to_19_units,asian_male_45_54,
                                       male_30_to_34,male_60_61,two_parents_father_in_labor_force_families_with_young_children,employed_transportation_warehousing_utilities,
                                       commuters_drove_alone,income_40000_44999,white_male_55_64,two_parents_in_labor_force_families_with_young_children,
                                       hispanic_male_55_64,commute_35_39_mins,employed_construction,amerindian_pop,income_per_capita,female_80_to_84,female_75_to_79,
                                       female_21,employed_science_management_admin_waste,male_21,households,occupied_housing_units,one_parent_families_with_young_children)

rf <-randomForest(deaths_norm ~ . , 
                  data=topUSDeathNormFeatures,
                  importance=TRUE,ntree=1000,na.action=na.exclude)
imp = importance(rf, type=1)
imp <- data.frame(predictors=rownames(imp),imp)
sorted = imp
sorted= sorted[order(sorted$X.IncMSE, decreasing = TRUE),] 
sorted
#write.csv(sorted, file = "sortedFeatureImportanceRandomForest-DeathNorm.csv", row.names = TRUE)

rf <-randomForest(cases_norm ~ . , 
                  data=topUSCovidNormFeatures,
                  importance=TRUE,ntree=1000,na.action=na.exclude)
imp = importance(rf, type=1)
imp <- data.frame(predictors=rownames(imp),imp)
sorted = imp
sorted= sorted[order(sorted$X.IncMSE, decreasing = TRUE),] 
sorted
#write.csv(sorted, file = "sortedFeatureImportanceRandomForest-CovidNorm.csv", row.names = TRUE)

#includes top 30 from random forest
topDeathNormFeatures = dplyr::select(cases_normalized, -Class, -total_pop, -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths,
                                     dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                     mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                     vacant_housing_units,white_male_45_54, median_age,vacant_housing_units_for_sale,commuters_by_public_transportation,
                                     commuters_by_bus, workplaces_percent_change_from_baseline, parks_percent_change_from_baseline,transit_stations_percent_change_from_baseline,
                                     armed_forces, amerindian_pop, owner_occupied_housing_units_median_value, median_year_structure_built,median_income,other_race_pop,
                                     high_school_diploma, group_quarters, black_including_hispanic,commuters_by_subway_or_elevated,less_than_high_school_graduate,
                                     high_school_including_ged)        

TopDeathDataTestScaled =dplyr::select(deathDataTestScaled,Class,
                                      dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                      mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                      vacant_housing_units,white_male_45_54, median_age,vacant_housing_units_for_sale,commuters_by_public_transportation,
                                      commuters_by_bus, workplaces_percent_change_from_baseline, parks_percent_change_from_baseline,transit_stations_percent_change_from_baseline,
                                      armed_forces, amerindian_pop, owner_occupied_housing_units_median_value, median_year_structure_built,median_income,other_race_pop,
                                      high_school_diploma, group_quarters, black_including_hispanic,commuters_by_subway_or_elevated,less_than_high_school_graduate,
                                      high_school_including_ged)
TopDeathDataTrainScaled =dplyr::select(deathDataTrainScaled,Class,
                                       dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                       mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                       vacant_housing_units,white_male_45_54, median_age,vacant_housing_units_for_sale,commuters_by_public_transportation,
                                       commuters_by_bus, workplaces_percent_change_from_baseline, parks_percent_change_from_baseline,transit_stations_percent_change_from_baseline,
                                       armed_forces, amerindian_pop, owner_occupied_housing_units_median_value, median_year_structure_built,median_income,other_race_pop,
                                       high_school_diploma, group_quarters, black_including_hispanic,commuters_by_subway_or_elevated,less_than_high_school_graduate,
                                       high_school_including_ged)
TopDeathDataTestScaled =dplyr::select(deathDataTestScaled,Class,
                                       dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                       mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                       vacant_housing_units,white_male_45_54, median_age,vacant_housing_units_for_sale,commuters_by_public_transportation,
                                       commuters_by_bus, workplaces_percent_change_from_baseline, parks_percent_change_from_baseline,transit_stations_percent_change_from_baseline,
                                       armed_forces, amerindian_pop, owner_occupied_housing_units_median_value, median_year_structure_built,median_income,other_race_pop,
                                       high_school_diploma, group_quarters, black_including_hispanic,commuters_by_subway_or_elevated,less_than_high_school_graduate,
                                       high_school_including_ged)
Top10DeathDataTrainScaled =dplyr::select(deathDataTrainScaled,Class,
                                       dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                       mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                       vacant_housing_units,white_male_45_54) 
Top10DeathDataTestScaled =dplyr::select(deathDataTestScaled,Class,
                                         dwellings_2_units,employed_agriculture_forestry_fishing_hunting_mining, different_house_year_ago_different_city,two_or_more_races_pop,
                                         mobile_homes, male_45_64_high_school,owner_occupied_housing_units_lower_value_quartile, percent_income_spent_on_rent,
                                         vacant_housing_units,white_male_45_54)
topCovidNormFeatures = dplyr::select(cases_normalized, -Class, -total_pop, -confirmed_cases, -delta, -county_fips_code, -geo_id, -state_fips_code, -state, -date,-gini_index, -county_name, -deaths,
                                     armed_forces,female_female_households,vacant_housing_units, hispanic_male_55_64,commute_90_more_mins,
                                     owner_occupied_housing_units_median_value,hispanic_male_45_54,median_rent,hispanic_pop,median_year_structure_built,
                                     asian_male_45_54,hispanic_any_race,two_or_more_races_pop,transit_stations_percent_change_from_baseline,parks_percent_change_from_baseline,
                                     male_45_64_grade_9_12,percent_income_spent_on_rent,median_age,median_income,
                                     four_more_cars,asian_pop,amerindian_pop,asian_including_hispanic,amerindian_including_hispanic,
                                     employed_agriculture_forestry_fishing_hunting_mining,income_per_capita,less_than_high_school_graduate)

TopCovidDataTestScaled =dplyr::select(CovidDataTestScaled,Class,
                                      armed_forces,female_female_households,vacant_housing_units, hispanic_male_55_64,commute_90_more_mins,
                                      owner_occupied_housing_units_median_value,hispanic_male_45_54,median_rent,hispanic_pop,median_year_structure_built,
                                      asian_male_45_54,hispanic_any_race,two_or_more_races_pop,transit_stations_percent_change_from_baseline,parks_percent_change_from_baseline,
                                      male_45_64_grade_9_12,percent_income_spent_on_rent,median_age,median_income,
                                      four_more_cars,asian_pop,amerindian_pop,asian_including_hispanic,amerindian_including_hispanic,
                                      employed_agriculture_forestry_fishing_hunting_mining,income_per_capita,less_than_high_school_graduate)    
Top10CovidDataTestScaled =dplyr::select(CovidDataTestScaled,Class,
                                        armed_forces,female_female_households,vacant_housing_units, hispanic_male_55_64,commute_90_more_mins,
                                        owner_occupied_housing_units_median_value,hispanic_male_45_54,median_rent,hispanic_pop,median_year_structure_built,asian_male_45_54)    
TopCovidDataTrainScaled =dplyr::select(CovidDataTrainScaled,Class,
                                       armed_forces,female_female_households,vacant_housing_units, hispanic_male_55_64,commute_90_more_mins,
                                       owner_occupied_housing_units_median_value,hispanic_male_45_54,median_rent,hispanic_pop,median_year_structure_built,
                                       asian_male_45_54,hispanic_any_race,two_or_more_races_pop,transit_stations_percent_change_from_baseline,parks_percent_change_from_baseline,
                                       male_45_64_grade_9_12,percent_income_spent_on_rent,median_age,median_income,
                                       four_more_cars,asian_pop,amerindian_pop,asian_including_hispanic,amerindian_including_hispanic,
                                       employed_agriculture_forestry_fishing_hunting_mining,income_per_capita,less_than_high_school_graduate)   
Top10CovidDataTrainScaled =dplyr::select(CovidDataTrainScaled,Class,
                                         armed_forces,female_female_households,vacant_housing_units, hispanic_male_55_64,commute_90_more_mins,
                                         owner_occupied_housing_units_median_value,hispanic_male_45_54,median_rent,hispanic_pop,median_year_structure_built,
                                         asian_male_45_54) 

#sets 1- 3 include mobility data, sets 4 and 5 do not
covidFeatureSet1train=TopCovidDataTrainScaled
covidFeatureSet1test= TopCovidDataTestScaled
covidFeatureSet2train=Top10CovidDataTrainScaled
covidFeatureSet2test=Top10CovidDataTestScaled 
#Note  grocery_and_pharmacy_percent_change_from_baseline and retail_and_recreation_percent_change_from_baseline
#were not in test and train datasets
covidFeatureSet3train = dplyr::select (CovidDataTrainScaled, Class, transit_stations_percent_change_from_baseline,residential_percent_change_from_baseline,
                                       workplaces_percent_change_from_baseline)
covidFeatureSet3test =dplyr::select (CovidDataTestScaled,Class, transit_stations_percent_change_from_baseline,residential_percent_change_from_baseline,
                                     workplaces_percent_change_from_baseline)
covidFeatureSet4train = dplyr::select (covidFeatureSet1train, -transit_stations_percent_change_from_baseline,-parks_percent_change_from_baseline)
covidFeatureSet4test =dplyr::select (covidFeatureSet1test, -transit_stations_percent_change_from_baseline,-parks_percent_change_from_baseline)


covidFeatureSet5train =dplyr::select (covidFeatureSet4train,Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)
covidFeatureSet5test = dplyr::select (covidFeatureSet4test,Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)

deathFeatureSet1train=TopDeathDataTrainScaled
deathFeatureSet1test = TopDeathDataTestScaled
deathFeatureSet2train =Top10DeathDataTrainScaled
deathFeatureSet2test = Top10DeathDataTestScaled
deathFeatureSet3train =dplyr::select (deathDataTrainScaled, Class, transit_stations_percent_change_from_baseline,residential_percent_change_from_baseline,
                                      workplaces_percent_change_from_baseline)
deathFeatureSet3test= dplyr::select (deathDataTestScaled, Class, transit_stations_percent_change_from_baseline,residential_percent_change_from_baseline,
                                     workplaces_percent_change_from_baseline)
deathFeatureSet4train = dplyr::select (deathFeatureSet1train, -workplaces_percent_change_from_baseline,-transit_stations_percent_change_from_baseline,-parks_percent_change_from_baseline)
deathFeatureSet4test =dplyr::select (deathFeatureSet1test , -workplaces_percent_change_from_baseline, -transit_stations_percent_change_from_baseline,-parks_percent_change_from_baseline)
deathFeatureSet5train =dplyr::select (deathFeatureSet4train,Class, group_quarters,employed_agriculture_forestry_fishing_hunting_mining, 
                                      different_house_year_ago_different_city,commuters_by_public_transportation, vacant_housing_units)
deathFeatureSet5test = dplyr::select (deathFeatureSet4test,Class, group_quarters,employed_agriculture_forestry_fishing_hunting_mining, 
                                      different_house_year_ago_different_city,commuters_by_public_transportation, vacant_housing_units)

covidtrainFeatureSets=c( list(covidFeatureSet1train),list(covidFeatureSet2train),list(covidFeatureSet3train),list(covidFeatureSet4train),list(covidFeatureSet5train) )
covidtestFeatureSets=c( list(covidFeatureSet1test),list(covidFeatureSet2test),list(covidFeatureSet3test),list(covidFeatureSet4test),list(covidFeatureSet5test) )
deathtrainFeatureSets=c( list(deathFeatureSet1train),list(deathFeatureSet2train),list(deathFeatureSet3train),list(deathFeatureSet4train),list(deathFeatureSet5train) )
deathtestFeatureSets=c( list(deathFeatureSet1test),list(deathFeatureSet2test),list(deathFeatureSet3test),list(deathFeatureSet4test),list(deathFeatureSet5test) )

# --------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Classification Models:
#
# Logistic Regression Deaths   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

LogDeathsModel <- glm(as.factor(Class) ~ . , data = deathDataTrainScaled, family = "binomial")
summary(LogDeathsModel)
# predict(LogDeathsModel, deathDataTestScaled)
pr <- predict(LogDeathsModel, deathDataTestScaled, type = "response")
round(pr, 2)
hist(pr, breaks=20)
table(actual=deathDataTestScaled$Class, predicted=pr>.5)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Logistic Regression Cases   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

LogCovidModel <- glm(as.factor(Class) ~ . , data = CovidDataTrainScaled, family = "binomial")
summary(LogCovidModel)
# predict(LogDeathsModel, deathDataTestScaled)
pr <- predict(LogCovidModel, CovidDataTestScaled, type = "response")
round(pr, 2)
hist(pr, breaks=20)
table(actual=CovidDataTestScaled$Class, predicted=pr>.5)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Multinomial Logistic Regression All Death Feature Sets   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training the multinomial models
j=0
multinomDeathModels = list()
for ( i in 1:length(deathtrainFeatureSets)){
   j = j+1
   multinomDeathModel <- multinom(Class ~., data =deathtrainFeatureSets[[i]])
   if (j ==1){
     multinomDeathModels = list(multinomDeathModel)
   }
   else{
     multinomDeathModels = c(multinomDeathModels,list(multinomDeathModel))
   }

   # Checking the model
   #summary(multinomDeathModel)
   #sort(coefficients(multinomDeathModel))
   #imp =caret::varImp(multinomDeathModel)
   #imp <- as.data.frame(imp)
   #imp <- data.frame(overall = imp$Overall,
    #              names   = rownames(imp))
   #imp[order(imp$overall,decreasing = T),]
}
## Predict and print Confusion Matrix for all models using Training sets
j=0
deathPredictCVs = list()
for ( i in 1:length(multinomDeathModels)){
    j = j+1
    PredictCV = predict(multinomDeathModels[[i]], newdata = deathtrainFeatureSets[[j]], type = "class",  na.action=na.pass)
    deathPredictCVs = c(deathPredictCVs, list(PredictCV))
}
length(deathPredictCVs)
#Confusion Matrix
j=0
deathConfusionMatrix = list()
for (i in 1:length(deathPredictCVs)){
  j= j+1
  tab1 = table(deathtrainFeatureSets[[j]]$Class, deathPredictCVs[[i]])
  deathConfusionMatrix = c(deathConfusionMatrix, list(tab1))
}  
length(deathConfusionMatrix)
for ( i in 1:length(deathConfusionMatrix) ){
  out=deathConfusionMatrix[[i]]
  print(out) 
  print(((deathConfusionMatrix[[i]][1,1]+deathConfusionMatrix[[i]][2,2]+deathConfusionMatrix[[i]][3,3])) /sum(deathConfusionMatrix[[i]]))
}
## Predict and print Confusion Matrix for all models using Test sets
j=0
deathPredictCVs = list()
for ( i in 1:length(multinomDeathModels)){
  j = j+1
  PredictCV = predict(multinomDeathModels[[i]], newdata = deathtestFeatureSets[[j]], type = "class",  na.action=na.pass)
  deathPredictCVs = c(deathPredictCVs, list(PredictCV))
}
length(deathPredictCVs)
#Confusion Matrix
j=0
deathConfusionMatrix = list()
for (i in 1:length(deathPredictCVs)){
  j= j+1
  tab1 = table(deathtestFeatureSets[[j]]$Class, deathPredictCVs[[i]])
  deathConfusionMatrix = c(deathConfusionMatrix, list(tab1))
}  
length(deathConfusionMatrix)
for ( i in 1:length(deathConfusionMatrix) ){
  out=deathConfusionMatrix[[i]]
  print(out) 
  print(((deathConfusionMatrix[[i]][1,1]+deathConfusionMatrix[[i]][2,2]+deathConfusionMatrix[[i]][3,3])) /sum(deathConfusionMatrix[[i]]))
}

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Multinomial Logistic Regression All Covid Feature Sets
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
j=0

multinomCovidModels = list()
for ( i in 1:length(covidtrainFeatureSets)){
  j = j+1
  multinomCovidModel <- multinom(Class ~., data =covidtrainFeatureSets[[i]])
  if (j ==1){
    multinomCovidModels = list(multinomCovidModel)
  }
  else{
    multinomCovidModels = c(multinomCovidModels,list(multinomCovidModel))
  }
} 
## Predict and print Confusion Matrix for all models using Training sets
j=0
covidPredictCVs = list()
for ( i in 1:length(multinomCovidModels)){
  j = j+1
  PredictCV = predict(multinomCovidModels[[i]], newdata = covidtrainFeatureSets[[j]], type = "class",  na.action=na.pass)
  covidPredictCVs = c(covidPredictCVs, list(PredictCV))
}
length(covidPredictCVs)
#Confusion Matrix
j=0
covidConfusionMatrix = list()
for (i in 1:length(covidPredictCVs)){
  j= j+1
  tab1 = table(covidtrainFeatureSets[[j]]$Class, covidPredictCVs[[i]])
  covidConfusionMatrix = c(covidConfusionMatrix, list(tab1))
}  
length(covidConfusionMatrix)
for ( i in 1:length(covidConfusionMatrix) ){
  out=covidConfusionMatrix[[i]]
  print(out) 
  print(((covidConfusionMatrix[[i]][1,1]+covidConfusionMatrix[[i]][2,2]+covidConfusionMatrix[[i]][3,3])) /sum(covidConfusionMatrix[[i]]))
}

## Predict and print Confusion Matrix for all models using Test sets
j=0
covidPredictCVs = list()
for ( i in 1:length(multinomCovidModels)){
  j = j+1
  PredictCV = predict(multinomCovidModels[[i]], newdata = covidtestFeatureSets[[j]], type = "class",  na.action=na.pass)
  covidPredictCVs = c(covidPredictCVs, list(PredictCV))
}
length(covidPredictCVs)
#Confusion Matrix
j=0
covidConfusionMatrix = list()
for (i in 1:length(covidPredictCVs)){
  j= j+1
  tab1 = table(covidtestFeatureSets[[j]]$Class, covidPredictCVs[[i]])
  covidConfusionMatrix = c(covidConfusionMatrix, list(tab1))
}  
length(covidConfusionMatrix)
for ( i in 1:length(covidConfusionMatrix) ){
  out=covidConfusionMatrix[[i]]
  print(out) 
  print(((covidConfusionMatrix[[i]][1,1]+covidConfusionMatrix[[i]][2,2]+covidConfusionMatrix[[i]][3,3])) /sum(covidConfusionMatrix[[i]]))
}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision Tree Deaths   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#na.exclude(deathDataTrainScaled$Class)
train_index <-createFolds(deathDataTrainScaled$Class, k =10)
ctreeFitDeath <- deathDataTrainScaled %>% train(Class ~ .,
                                                method = "ctree",
                                                data = .,
                                                tuneLength = 5,
                                                trControl = trainControl(method = "cv", indexOut = train_index))
ctreeFitDeath
dev.off()
plot(ctreeFitDeath$finalModel)

# Checking the model
summary(ctreeFitDeath)

prCtreeDeathDataTestScaled <- predict(ctreeFitDeath, deathDataTestScaled)
summary(prCtreeDeathDataTestScaled)
as.factor(prCtreeDeathDataTestScaled)
ctreeconfusionMatrixDeaths <- confusionMatrix(as.factor(deathDataTestScaled$Class), prCtreeDeathDataTestScaled)
ctreeconfusionMatrixDeaths

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision Tree Cases  
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_index <-createFolds(CovidDataTrainScaled$Class, k =10)
ctreeFitCovid <- CovidDataTrainScaled %>% train(Class ~ .,
                                                method = "ctree",
                                                data = .,
                                                tuneLength = 5,
                                                trControl = trainControl(method = "cv", indexOut = train_index))
ctreeFitCovid
dev.off()
plot(ctreeFitCovid$finalModel)

prCtreeCovidDataTestScaled <- predict(ctreeFitCovid, CovidDataTestScaled)
summary(prCtreeCovidDataTestScaled)
as.factor(prCtreeCovidDataTestScaled)
ctreeconfusionMatrixCovid <- confusionMatrix(as.factor(CovidDataTestScaled$Class), prCtreeCovidDataTestScaled)
ctreeconfusionMatrixCovid

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# K-nearest Neighbor Deaths   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_index <-createFolds(deathDataTrainScaled$Class, k =10)
knnFitDeath <- deathDataTrainScaled %>% train(Class ~ ., 
                                              method = "knn", 
                                              data= .,
                                              preProcess = "scale", 
                                              tuneLength = 5,
                                              tuneGrid=data.frame(k=1:10),
                                              trControl = trainControl(method = "cv", indexOut = train_index))
knnFitDeath

knnFitDeath$finalModel

prKNNDeathDataTestScaled <- predict(knnFitDeath, deathDataTestScaled)
summary(prKNNDeathDataTestScaled)
as.factor(prKNNDeathDataTestScaled)
KNNconfusionMatrixDeaths <- confusionMatrix(as.factor(deathDataTestScaled$Class), prKNNDeathDataTestScaled)
KNNconfusionMatrixDeaths

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# K-nearest neighbor Cases   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_index <-createFolds(CovidDataTrainScaled$Class, k =10)
knnFitCase <- CovidDataTrainScaled %>% train(Class ~ ., 
                                             method = "knn", 
                                             data= .,
                                             preProcess = "scale", 
                                             tuneLength = 5,
                                             tuneGrid=data.frame(k=1:10),
                                             trControl = trainControl(method = "cv", indexOut = train_index))
knnFitCase

knnFitCase$finalModel

prKNNCovidDataTestScaled <- predict(knnFitCase, CovidDataTestScaled)
summary(prKNNCovidDataTestScaled)
as.factor(prKNNCovidDataTestScaled)
KNNconfusionMatrixCovid <- confusionMatrix(as.factor(CovidDataTestScaled$Class), prKNNCovidDataTestScaled)
KNNconfusionMatrixCovid

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Support Vector Machine Deaths   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Training the support vector machine (SVM) model
svmfit <- svm(formula = Class ~ ., data = deathDataTrainScaled, cross=10, type = 'C-classification', kernel = 'linear')
print(svmfit)

# Checking the model
summary(svmfit)

actual <- as.factor(deathDataTestScaled$Class)

# Predicting the Test set results
svm_pred <- predict(svmfit, newdata = deathDataTestScaled)
summary(svm_pred)

confusionMatrix(actual, svm_pred)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Support Vector Machine Cases  
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Training the support vector machine (SVM) model
svmfit <- svm(formula = Class ~ ., data = CovidDataTrainScaled, cross=10, type = 'C-classification', kernel = 'linear')
print(svmfit)

# Checking the model
summary(svmfit)

actual <- as.factor(CovidDataTestScaled$Class)

# Predicting the Test set results
svm_pred <- predict(svmfit, newdata = CovidDataTestScaled)
summary(svm_pred)

confusionMatrix(CovidDataTest$Class, svm_pred)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Artificial Neural Network Deaths   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Work on parameters of hidden layers of NN
n <- names(deathDataTrainScaled)
f <- as.formula(paste("Class ~", paste(n[! n %in% "Class"], collapse = " + ")))
# Training the artificial neural network (ANN) model
nn <- neuralnet(f, data = deathDataTrainScaled, hidden = c(5, 3), linear.output = T)
print(nn)
# Checking the model
summary(nn)

# Predicting the Test set results
nn_pred = predict(nn, newdata = deathDataTestScaled)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Artificial Neural Network Cases   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Work on parameters of hidden layers of NN
n <- names(CovidDataTrainScaled)
f <- as.formula(paste("Class ~", paste(n[! n %in% "Class"], collapse = " + ")))
# Training the artificial neural network (ANN) model
nn <- neuralnet(f, data = CovidDataTrainScaled, hidden = c(5, 3), linear.output = T)
print(nn)
# Checking the model
summary(nn)

# Predicting the Test set results
nn_pred = predict(nn, newdata = CovidDataTestScaled)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
