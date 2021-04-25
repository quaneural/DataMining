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
library("relaimpo")
library("randomForest")

# Establish connection to bigrquery database and query data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con <- dbConnect(
  bigrquery::bigquery(),
  project = "My First Project",
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Begin Feature Importance Analysis - Bridget  ------------------------------------------------------------------------
#     CHI SQUARE AND CFS
#  Find most important features across the US   ------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
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

#   End Feature Importance Analysis  -----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

