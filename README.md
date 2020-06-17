# COVID-19 US States Feature Set
We create a dataset (Main.xlsx) of US states and several features that may potentially influence the infected and death counts due to COVID-19. Below we discuss the source and description of the different features.

1. Gross Domestic Product ($million) for US states [1] (filename: source/GDP.xlsx, feature name: GDP)

2. Distance from one state to another (is not the actual distance in miles) but the euclidean distance between their lat-long coordinates [2] (filename: source/Data_distance.xlsx, feature name: d(State))

3. Gender feature(s) is a fraction of total population representing male and female [3] (filename: source/Data_gender.csv, feature name: Male, Female)

4. Ethnicity feature(s) are the fraction of total population representing white, black, hispanic and asian (we leave out other smaller ethnic groups) [4] (filename: source/Data_ethnic.csv, feature name: White, Black, Hispanic and Asian)

5. Healthcare index is measured by the Agency for Healthcare Research and Quality (AHRQ) on the basis of (1) type of care (such as preventive or chronic), (2) setting of care (such as nursing homes or hospitals), and (3) clinical areas (such as care for patients with cancer or diabetes) [5] (filename: source/Data_health.xlsx, feature name: Health)

6. Homeless feature is the number of homeless individuals (normalized by the population of a state) [6] (filename: source/Data_homeless.xlsx, feature name: Homeless, Normalized Homeless)

7. Total cases (and death) of COVID-19 tested positive individuals (normalized by the population of a state) [7] (filename: source/Data_covid_total.xlsx, feature name: Total Cases, Normalized cases, Total Death, Normalized deaths) 

8. Infected score (scale 0 - 5) and death score (0 - 6) is obtained by normalizing total cases and deaths of a state by the sum of infected and death across all states (feature name: Infected Score, Death Score)

9. Death-to-Infected discrepancy is a feature measuring Death max( Score - Infected Score, 0) (feature name: Death-Infected)

10. Lockdown type is a feature capturing the type of lockdown in a given state [8] (filename: source/Data_lockdown.csv, feature name: Lockdown)

11. Day of lockdown is a feature capturing the number of difference between 1st of January, 2020 to the date of lockdown [8] (filename: source/Data_lockdown.csv, feature name: Day Lockdown)

12. Population density is the ratio between the population and area of a region [9] (filename: source/Data_population.csv, feature name: Population, Area, Population Density)

11. Business of airport measures passenger traffic normalized the business across all states [10] (filename: source/Data_airport.xlsx, feature name: Busy airport score, Normalized busy airport)

12.

References

[1] https://worldpopulationreview.com/states/gdp-by-state/

[2] https://en.wikipedia.org/wiki/List_of_geographic_centers_of_the_United_States#Updated_list

[3] https://www.kff.org/other/state-indicator/distribution-by-gender/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

[4] https://www.kff.org/other/state-indicator/distribution-by-raceethnicity/?dataView=0&currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

[5] https://www.ahrq.gov/data/infographics/state-compare-text.html

[6] https://www.hudexchange.info/resource/3300/2013-ahar-part-1-pit-estimates-of-homelessness/

[7] https://www.cdc.gov/covid-data-tracker/#testing

[8] https://www.kaggle.com/lin0li/us-lockdown-dates-dataset

[9] https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-detail.html

[10] https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States

[11] 
