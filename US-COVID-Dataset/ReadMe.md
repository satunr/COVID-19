# COVID-19 US States and New York City Feature Set
We create a dataset (Main.xlsx) of US states and several features that may potentially influence the infected and death counts due to COVID-19. Below we discuss the source and description of the different features.

1. Gross Domestic Product (in terms of million US dollars) for US states[1] (filename:  source/GDP.xlsx, feature name:  GDP).

2. Distance from one state to another (is not measured in miles but the euclidean distance between their latitude-longitude  coordinates[2]) (filename: source/Data_distance.xlsx, feature name: d(state1,state2))

3. Gender feature(s) is a fraction of total population representing the male and female individuals[3] (filename: source/Data_gender.csv, feature name:  Male, Female)

4. Ethnicity feature(s) are the fraction of total population representing white, black, hispanic and Asian individuals (we leave out other smaller ethnic groups)[4] (filename: source/Data_ethnic.csv, feature name: White, Black, Hispanic and Asian)

5. Healthcare index is measured by the Agency for Healthcare Research and Quality (AHRQ) on the basis of (1) type of care (such as preventive or chronic), (2) setting of care (such as nursing homes or hospitals), and (3) clinical areas (such as care for patients with cancer or diabetes)[5] (filename: source/Data_health.xlsx, feature name: Health)

6. Homeless feature is the number of homeless individuals (normalized by the population of a state) [6] (filename: source/Data_homeless.xlsx, feature name: Homeless, Normalized Homeless)

7. Total cases (and  deaths) of COVID-19 is the number of individuals tested positive and dead (normalized by the population of a state)[7] (filename: source/Data_covid_total.xlsx, feature name: Total cases, Normalized cases, Total death, Normalized deaths).

8. Infected score and death score is obtained by rounding normalized total cases and deaths to a discrete value between 0 and 6 (feature name: Infected Score, Death Score).

9. Death-to-Infected discrepancy is a feature measuring impact of death, calculated as:

max(Death Score - Infected Score, 0) 

(feature name: Death-Infected)

10. Lockdown type is a feature capturing the type of lockdown (shelter in place: 1 and stay at home: 2) in a given state[7,8] (filename:source/Data_lockdown.csv, feature name: Lockdown)

11. Day of lockdown captures the difference in days between 1st January 2020 to the date of imposition of lockdown in a region[10] (filename: source/Data_lockdown.csv, feature name: Day Lockdown)

12. Population density is the ratio between the population and area of a region[10] (filename: source/Data_population.csv, feature name: Population, Area, Population Density).

13. Traffic/activity of airport measures the passenger traffic (also normalized by the total traffic across all the states of USA[11] (filename: source/Data_airport.xlsx, feature name: Busy airport score, Normal-ized busy airport)

14. Age groups (0 - 85+) in brackets of 4 year (also normalized by total population)[10] (filename: source/Data_age.xlsx, feature name: age_to_ Norm_to_, e.g. age4to8); we later group them in brackets of 20 for the purposes of analysis.

15. Peak infected (and peak death) measures the duration between first date of infection and date of daily infected (and death) peaks[10] (feature name: Peak Infected, Peak Death).

16. Testing measures the number of individuals tested for COVID-19 (total number before and after imposition of lockdown[8,12] (filename: source/Data_testing.xlsx, feature name: Testing, Pre-lockdown testing, Post-lockdown testing)

17. Pre- and post-infected and death count measures the number of individuals infected and dead before and after lockdown dates (feature name:Testing, Pre-infected count, Pre-death count, Post-infected count, Post-death count).

18. Days between first infected and lockdown date (feature name: First-Inf-Lockdown)

19. Percentage change in GDP between 2019-2020 (filename:  source/GDP.xlsx, feature name:  GDP).

#--------------------------------------------------------------------

New York City dataset (NYC_dist_mob.xlsx) captures the mobility and COVID-19 data for the 5 boroughs of NYC.

1. Mobility data (based on traffic volume counts collected by DOT for New York Metropolitan Transportation Council (NYMTC[13]) shows the number of trips from one borough to another

2. COVID-19 data shows the number of COVID-19 infected and death counts for each borough [14].

#--------------------------------------------------------------------

The New York Times reported a state-wise list of businesses that are shut down as per state laws) to prevent social contact and curb infection spread [15]. 


The US Bureau of Labor Statistics put out a summary of the number of individuals in different ethnic groups hired by each job sector in 2019 [16].


The US Census Bureau report presents an yearly breakdown of the population on the basis of age, sex, race, family arrangement, income and poverty, etc. [17].

#--------------------------------------------------------------------

References

[1] https://worldpopulationreview.com/states/gdp-by-state/

[2] https://en.wikipedia.org/wiki/List_of_geographic_centers_of_the_United_States#Updated_list

[3] https://www.kff.org/other/state-indicator/distribution-by-gender/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

[4] https://www.kff.org/other/state-indicator/distribution-by-raceethnicity/?dataView=0&currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

[5] https://www.ahrq.gov/data/infographics/state-compare-text.html

[6] https://www.hudexchange.info/resource/3300/2013-ahar-part-1-pit-estimates-of-homelessness/

[7] https://www.cdc.gov/covid-data-tracker/#testing

[8] https://www.worldometers.info/coronavirus/country/us/

[9] https://www.kaggle.com/lin0li/us-lockdown-dates-dataset

[10] https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-detail.html

[11] https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States

[12] https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/previous-testing-in-us.html

[13] https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts-2012-2013-/p424-amsu

[14] https://data.beta.nyc/pages/nyc-covid19

[15] https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html

[16] https://www.bls.gov/cps/cpsaat11.htm

[17] https://www.census.gov/quickfacts/fact/table/US/PST045219

[18] https://www.bea.gov/news/2021/gross-domestic-product-state-4th-quarter-2020-and-annual-2020-preliminary

