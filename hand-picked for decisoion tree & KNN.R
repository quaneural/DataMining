# set5

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
CovidDataTrain2= dplyr::select(CovidDataTrain, Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)
CovidDataTrainScaled = CovidDataTrain2 %>% dplyr::mutate_if(is.numeric, scale)
CovidDataTest2=dplyr::select(CovidDataTest,  Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)
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
deathDataTrain2= dplyr::select(deathDataTrain, Class, group_quarters,employed_agriculture_forestry_fishing_hunting_mining, 
                               different_house_year_ago_different_city,commuters_by_public_transportation, vacant_housing_units)
deathDataTrainScaled = deathDataTrain2 %>% dplyr::mutate_if(is.numeric, scale)
deathDataTest2=dplyr::select(deathDataTest, Class, group_quarters,employed_agriculture_forestry_fishing_hunting_mining, 
                             different_house_year_ago_different_city,commuters_by_public_transportation, vacant_housing_units)
deathDataTestScaled=deathDataTest2%>% dplyr::mutate_if(is.numeric, scale)


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

# another way
ctreeFitDeath <- deathDataTrainScaled %>% train(Class ~ .,
                                                method = "rpart",
                                                data = .,
                                                control = rpart.control(maxdepth = 5),
                                                tuneGrid = data.frame(cp = 0.01),
                                                parms = list(split ="information"))
ctreeFitDeath
rpart.plot(ctreeFitDeath$finalModel, extra = 2, cex = 0.65)
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

# another way
ctreeFitCovid <- CovidDataTrainScaled %>% train(Class ~ .,
                                                method = "rpart",
                                                data = .,
                                                control = rpart.control(maxdepth = 5),
                                                tuneGrid = data.frame(cp = 0.01),
                                                parms = list(split ="information"))
ctreeFitDeath
rpart.plot(ctreeFitCovid$finalModel, extra = 2, cex = 0.55)

prCtreeCovidDataTestScaled <- predict(ctreeFitCovid, CovidDataTestScaled)
summary(prCtreeCovidDataTestScaled)
as.factor(prCtreeCovidDataTestScaled)
ctreeconfusionMatrixCovid <- confusionMatrix(as.factor(CovidDataTestScaled$Class), prCtreeCovidDataTestScaled)
ctreeconfusionMatrixCovid

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# K-nearest Neighbor Deaths (no map-plotting)   
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


# Map
counties <- as_tibble(map_data("county"))
counties <- counties %>% 
  dplyr::rename(c(county = subregion, state = region)) %>%
  dplyr::mutate(state = state.abb[match(state, tolower(state.name))]) %>%
  dplyr::select(state, county, long, lat, group)
counties  

CovidDataTrain2= dplyr::select(CovidDataTrain, county_name, state, Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)
CovidDataTrainScaled = CovidDataTrain2 %>% dplyr::mutate_if(is.numeric, scale)
CovidDataTest2=dplyr::select(CovidDataTest,  county_name, state, Class, armed_forces,employed_agriculture_forestry_fishing_hunting_mining,four_more_cars,income_per_capita,less_than_high_school_graduate)
CovidDataTestScaled=CovidDataTest2%>% dplyr::mutate_if(is.numeric, scale)


counties_test <- counties %>% left_join(CovidDataTestScaled %>% 
                                          dplyr::mutate(county = county_name %>% str_to_lower() %>% 
                                                          str_replace('\\s+county\\s*$', '')))

ggplot(counties_test, aes(long, lat)) + 
  geom_polygon(aes(group = group, fill = Class), color = "black", size = 0.1) + 
  coord_quickmap() + 
  scale_fill_manual(values = c('HIGH' = 'red', 'MEDIUM' = 'yellow', 'LOW' = 'green'))
