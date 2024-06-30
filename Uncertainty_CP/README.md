Advancing Infection Profiling under Data Uncertainty through Contagion Potential
================================================================================

Below are the descriptions of the scripts.

### CP_OPT_BULK.py ###
<p align="justify"> This script implements a stochastic SIRS (Susceptible-Infected-Recovered-Susceptible) model to simulate the spread of an infectious disease over a population of one million individuals. The model calculates daily updates of the number of susceptible, infected, and recovered individuals based on given parameters (such as infection and recovery rates) and stores these updates in lists. The code also involves sampling to estimate new infections and recoveries over a specified duration and uses optimization techniques to adjust parameters for best fit to the model. Confidence intervals are computed for these estimates, and the results are compared to a bulk dataset to evaluate the model's accuracy. Finally, it stores and loads the simulation and optimization results using pickle.</p>

### CP_OPT_COUNTRY.py ###

<p align="justify"> This code is designed to analyze time-series epidemiological data for a geographical region, e.g. country, to infer the Contagion Potential (CP) with high confidence. It reads COVID-19 infection and recovery data from a CSV file, processes the cumulative and daily counts of infections and recoveries, and calculates key epidemiological parameters such as the infection rate (NI), recovery rate (NR), and current infected count ($I_0$). It then performs multiple iterations of statistical sampling and optimization to estimate parameters like the contact rate (C), gamma (recovery rate), and reproduction number ($R_0$), applying confidence intervals to assess the variability and reliability of the CP estimates. The optimization process ensures that the model parameters are consistent with observed data, aiming to provide a robust measure of CP over the given period. Finally, the results are compared to bulk data, and the consistency and fluctuations of the CP estimates are analyzed and saved for further examination.</p>

### dir_parallel2d.py ###

Confidence interval estimation considering complete CP information of the population varying mobility models, sample sizes, and strain infectivity
### inflate_upd.py ###
Confidence interval estimation considering incomplete CP information of the population and accounting for the adjustment in CP calculation
### CI_opt_germany.py and CI_opt_italy.py ###
Confidence interval estimation on Bulk Germany and Italy dataset
### Outbreak.py ###
Examining CI estimation in the event of an outbreak
### SIM_HCMM.py ###
Confidence interval estimation considering incomplete CP information of the population for an HCMM mobility model and using simple mean calculation for Grid CP
### SIM_IPW_SimpMean_Loc.py ###
Confidence interval estimation considering incomplete CP information of the population for localized mobility with inverse probability weighting for Grid CP calculation
### entropy.py ###
To calculate the entropy in localized and HCMM mobility simulations
### cov_data_italy_upd.csv, cov_data_germany_upd.csv
Time-series population-level epidemiological data of Italy (1 Jan 2022 - 13 Nov 2022) and Germany (1 Jan 2022 - 30 June 2022)
