# Pedal Perfect : Dynamic Demand Forecasting and Optimal Bike Allocation

# Executive Summary

## Business

Capital Bikeshare operates as a public bicycle sharing system, providing a convenient and eco-friendly transportation option to residents and visitors in the Washington, DC metropolitan area and surrounding cities. The system allows users to rent bicycles for short trips, typically for durations ranging from a few minutes to a few hours, and then return them to any designated station within the network.

Capital Bikeshare was established in September 2010 as a public-private partnership between the District of Columbia Department of Transportation (DDOT), Arlington County, Virginia, and Alta Bicycle Share, a private company specializing in bike-sharing systems. In 2013, Alta Bicycle Share was acquired by the bike-sharing company Motivate, which later became a subsidiary of Lyft. Today, Lyft manages the day-to-day operations of Capital Bikeshare, but the system is owned by the jurisdictions it serves, including Washington, DC, Arlington County, the City of Alexandria, Montgomery County, and Fairfax County. Since its inception, Capital Bikeshare has expanded its network to over 700 stations and more than 5,400 bicycles, serving millions of riders each year in the DC metropolitan area.

## Conundrum

Capital Bikeshare faces several challenges when determining the number of bikes and docks to make available at the start of each day to meet demand. One challenge is predicting the fluctuating demand patterns accurately. Factors such as weather, special events, and time of day can significantly impact bike usage, making it challenging to anticipate demand levels accurately. Capital Bikeshare must use historical data and predictive modeling techniques to forecast demand effectively.

Another challenge is ensuring optimal bike and dock availability at each station throughout the day. Capital Bikeshare needs to balance the distribution of bikes and docks across its network to meet demand at popular stations while avoiding overcrowding or underutilization at others. This requires real- time monitoring and adjustment of bike and dock allocations based on usage patterns and station capacity.

## Business Case

**Data-Driven Decision Making**: Machine Learning can provide Capital Bikeshare with valuable insights into customer behavior and usage patterns at the GWSB station. This can help the company make informed decisions about pricing, promotions, and other aspects of its business.

**Improved Customer Experience**: By accurately predicting the demand for bikes and docks at the GWSB station, Capital Bikeshare can ensure that there are enough bikes and docks available to meet customer needs. This can help reduce wait times and increase customer satisfaction.

**Increased Revenue**: By better predicting demand, Capital Bikeshare can potentially increase its revenue by attracting more customers to use its services. This can be particularly beneficial during peak times when demand is high.

## Goal

The primary goal of our project is to develop a series of Machine Learning models and recommend the best model to help Capital Bikeshare optimize the deployment of bikes and docks at the GWSB docking station. By leveraging historical ride records and weather information, along with advanced Machine Learning techniques, we aim to provide recommendations that will enhance the efficiency of bike and dock allocation. This, in turn, will improve the overall user experience for Capital Bikeshare customers.

## Approach

Our approach involves analyzing historical usage data of the GWSB docking station, including factors such as time of day, day of week, weather conditions, and special events. By identifying patterns and trends in this data, we aim to develop a predictive model that can forecast the demand for bikes and docks at different times.

To achieve this, we will use a variety of Machine Learning algorithms, including Linear Regression, K- Nearest Neighbors (KNN), Ridge Regression, Lasso, and ElasticNet. These algorithms will help us identify the key factors influencing bike and dock usage patterns. Additionally, we will employ techniques such as feature engineering and model evaluation to ensure the accuracy and reliability of our predictions.

# Exploratory Data Analysis

The process began by filtering the metadata to include only rides that both originated from and ended at the 22nd Street station. This filtering step ensured that the subsequent analysis focused exclusively on the rides directly involving this station, providing a more precise examination of its usage patterns. Subsequently, the rides were aggregated to determine the count of bike pickups and drop-offs.

Pearson’s correlation filtering between features: We generated a correlation matrix heatmap to visualize the relationships between the x-features. If variables were highly correlated, suggesting multicollinearity, we chose to exclude them. The feature filtering process involved selecting only one variable from pairs with a correlation coefficient exceeding 0.8, effectively reducing the impact of multicollinearity. Guided by the results of this analysis, we dropped the following features: 'feelslikemin', 'temp', 'dew', 'feelslikemax', 'solarenergy', 'uvindex', 'tempmin', 'windspeed', and 'feelslike'. Please refer to Figure 1.1 in the Appendix for details.

Some of the features that were still under consideration underwent additional analysis using Box Plots to determine if they violated certain assumptions of the general linear model. We specifically examined the data for qualities such as symmetry, tight grouping, and skewness. Based on this analysis, we decided to eliminate 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth', 'visibility', and 'severerisk' due to their distribution being noticeably different from the other features.
Upon merging the weather features (that remained in scope upon the previous exclusion) with ‘pickup’ and ‘drop off’ counts, the entire dataset was further subjected to Pearson correlation analysis to see how the target variables are related to the independent variables. The Mutual Info Regression package was used to perform this analysis. As observed in the heatmap, ‘cloudcover’, ‘humidity’, and ‘solarradiation’ show a noticeable correlation with daily pickups, indicating a potential relationship between these factors and bike usage patterns. Conversely, ‘moonphase’ and ‘winddir’ exhibit very low correlation coefficients with daily pickups. However, before deciding to remove ‘moonphase’ and ‘winddir’ from the analysis, it is important to consider the possibility that their low correlation could be attributed to a non-linear relationship with daily ‘pickups’. Therefore, it is necessary to employ a feature selection method that does not assume a linear functional form between the target variable and the features.

Upon further examination, ‘moonphase’ was found to have a mutual information score of 0 with daily pickups, indicating no significant predictive value. As a result, ‘moonphase’ will be excluded from the features used to predict daily pickups. However, ‘winddir’ despite its low correlation coefficient, has a non-zero Mutual Information score (refer Table 1.1), suggesting that it may still contain valuable predictive information and should therefore be retained in the analysis.

On the other hand, ‘cloudcover’, ‘humidity’, and ‘solarradiation’ exhibit noticeable correlations with drop- offs, suggesting a potential relationship between these factors and the number of bikes being returned. In contrast, ‘moonphase’, ‘winddir’, and ‘windgust’ show very low correlation coefficients with daily drop- offs. However, before deciding to remove ‘moonphase’, ‘winddir’, and ‘windgust’ from the analysis, it is important to consider the possibility that their low correlation could be attributed to a non-linear relationship with daily drop-offs. Therefore, it is necessary to use a feature selection method that does not assume a linear functional form between the target variable and the features.

Upon further investigation, ‘winddir’, and ‘windgust’ were found to have non-zero mutual information scores (refer Table 1.2) with daily drop-offs, indicating that they may still contain valuable predictive information. Therefore, these variables will be retained in the analysis.

Using Mutual Information, which measures the dependence between two variables, we can analyze both categorical and continuous variables. However, before using Mutual Information with categorical variables, they need to be numerically encoded. The value of the mutual information indicates the predictive power of a feature; a higher value suggests a stronger predictive relationship. Features with a mutual information score of 0 are considered largely independent and have no relationship with the target variable. Importantly, Mutual Information can detect non-linear relationships between variables.
