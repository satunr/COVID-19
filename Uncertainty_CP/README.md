Advancing Infection Profiling under Data Uncertainty through Contagion Potential
================================================================================

### Overview of the Framework ###

(Abstract to be added.)


### Module Description ###  

***CP_OPT_BULK.py*** <p align="justify"> This script implements a stochastic SIRS (Susceptible-Infected-Recovered-Susceptible) model to simulate the spread of an infectious disease over a population of one million individuals. The model calculates daily updates of the number of susceptible, infected, and recovered individuals based on given parameters (such as infection and recovery rates) and stores these updates in lists. The code also involves sampling to estimate new infections and recoveries over a specified duration and uses optimization techniques to adjust parameters for best fit to the model. Confidence intervals are computed for these estimates, and the results are compared to a bulk dataset to evaluate the model's accuracy. Finally, it stores and loads the simulation and optimization results using pickle.</p>

***CP_OPT_COUNTRY.py***
<p align="justify"> This script is designed to analyze time-series epidemiological data for a geographical region, e.g. country, to infer the Contagion Potential (CP) with high confidence. It reads COVID-19 infection and recovery data from a CSV file, processes the cumulative and daily counts of infections and recoveries, and calculates key epidemiological parameters such as the infection rate (NI), recovery rate (NR), and current infected count ($I_0$). It then performs multiple iterations of statistical sampling and optimization to estimate parameters like the contact rate (C), gamma (recovery rate), and reproduction number ($R_0$), applying confidence intervals to assess the variability and reliability of the CP estimates. The optimization process ensures that the model parameters are consistent with observed data, aiming to provide a robust measure of CP over the given period. Finally, the results are compared to bulk data, and the consistency and fluctuations of the CP estimates are analyzed and saved for further examination. </p>

***SIM_SPATIAL.py***
<p align="justify"> This script simulates the spread of an infectious disease in a population using a SEIRD (Susceptible, Exposed, Infected, Recovered, Deceased) model within a defined grid-based area. It initializes the population with specific statuses and locations, then iterates through a daily simulation of infection dynamics. The simulation updates the location and status of each individual based on their interactions and probabilities of infection, recovery, and death. It calculates contagion potentials (CPs) and uses statistical methods to estimate confidence intervals for these CPs. The code supports different modes of infection spread, such as random mixing or superspreader events, and uses multiprocessing to optimize performance. Results are periodically saved for analysis, including daily mean CPs, infection counts, and confidence intervals. </p>

***CP_INCOMPLETE.py***
<p align="justify"> The script simulates the spread of a disease within a population, with the primary aim of inferring Contagion Potential (CP) from incomplete epidemiological information. It models the movements and interactions of individuals in a spatial environment, considering factors like home locations, friendship networks, and infection probabilities. Using the SEIRD (Susceptible-Exposed-Infected-Recovered-Dead) model, the simulation tracks disease dynamics over time, adjusting individual statuses based on CP and infection probabilities. It employs sampling methods to infer CP and evaluates whether the real infection rate falls within a calculated confidence interval, continuously adjusting factors and logging results for comprehensive analysis. </p>

***ENTROPY.py***
<p align="justify"> The provided code calculates and compares the entropy of two different mixing matrices to measure the randomness in local mobility. Entropy is used as a metric to evaluate social mixing, with lower entropy indicating lower social mixing and consequently poorer estimation of Contagion Potential (CP) from sampled data. The code loads two mixing matrices from pickle files, computes the entropy for each row of these matrices, and then calculates the mean entropy for each matrix, providing an overall measure of randomness in the population's movements and interactions. </p>

***PARALLEL_SPATIAL_CP.py***
<p align="justify"> This code simulates a Susceptible-Exposed-Infectious-Recovered-Dead (SEIRD) model using different modes for generating locations and calculating interactions among individuals. It includes functions for setting up initial conditions, calculating spatial interactions based on proximity, updating individual statuses over time according to epidemiological parameters like transmission rates and recovery probabilities, and sampling infected cases to estimate confidence intervals. It employs multiprocessing for efficiency in computing pairwise interactions and utilizes various input data sources such as home percentages, friendship matrices, and geographical grid definitions. The model tracks daily status counts and mean Contagion Potentials (CPs) across multiple iterations, saving results to files for further analysis. </p>

***CP_BIAS_IPW.py***
<p align="justify"> In this script, inverse probability weighting (IPW) is strategically employed to counteract biases inherent in sample collection methods. By adjusting the contribution of each individual's contagion potential (CP) based on their likelihood of being sampled, IPW enhances the accuracy of CP predictions. This approach is crucial in epidemiological simulations where realistic estimations of infectious disease spread are paramount. </p>

### DATASETS ###
We consider population-level epidemiological data of the daily COVID cases in Germany, Italy, and Austria between January 1, 2022, and June 20, 2022, obtained from Our World in Data [1]. This dataset includes cumulative positive cases, cumulative deceased cases, cumulative recovered cases, current positive cases, hospitalization figures, intensive care data, etc., categorized by date and region within each country. 

### DEPENDENCIES ###

**pip install _package_**

package = {scipy, numpy, matplotlib, pickle-mixin, scipy, scikit-learn, pathos}


Time-series population-level epidemiological data of Italy (1 Jan 2022 - 13 Nov 2022) and Germany (1 Jan 2022 - 30 June 2022)



### REFERENCES ### 

[1] Our world in data. https://ourworldindata.org/covid-cases, 2022.

