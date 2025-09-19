# Pedal Perfect : Dynamic Demand Forecasting and Optimal Bike Allocation

# Executive Summary

## Business

Capital Bikeshare operates as a public bicycle sharing system, providing a convenient and eco-friendly transportation option to residents and visitors in the Washington, DC metropolitan area and surrounding cities. The system allows users to rent bicycles for short trips, typically for durations ranging from a few minutes to a few hours, and then return them to any designated station within the network.

Capital Bikeshare was established in September 2010 as a public-private partnership between the District of Columbia Department of Transportation (DDOT), Arlington County, Virginia, and Alta Bicycle Share, a private company specializing in bike-sharing systems. In 2013, Alta Bicycle Share was acquired by the bike-sharing company Motivate, which later became a subsidiary of Lyft. Today, Lyft manages the day-to-day operations of Capital Bikeshare, but the system is owned by the jurisdictions it serves, including Washington, DC, Arlington County, the City of Alexandria, Montgomery County, and Fairfax County. Since its inception, Capital Bikeshare has expanded its network to over 700 stations and more than 5,400 bicycles, serving millions of riders each year in the DC metropolitan area.

## Conundrum

Capital Bikeshare faces several challenges when determining the number of bikes and docks to make available at the start of each day to meet demand. One challenge is predicting the fluctuating demand patterns accurately. Factors such as weather, special events, and time of day can significantly impact bike usage, making it challenging to anticipate demand levels accurately. Capital Bikeshare must use historical data and predictive modeling techniques to forecast demand effectively.

Another challenge is ensuring optimal bike and dock availability at each station throughout the day. Capital Bikeshare needs to balance the distribution of bikes and docks across its network to meet demand at popular stations while avoiding overcrowding or underutilization at others. This requires real- time monitoring and adjustment of bike and dock allocations based on usage patterns and station capacity.

## Business Case

**Data-Driven Decision Making**: Machine Learning can provide Capital Bikeshare with valuable insights into customer behavior and usage patterns at the George Washington School of Business (GWSB) station. This can help the company make informed decisions about pricing, promotions, and other aspects of its business.

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

# Machine Learning Models
To identify the optimal set of features for accurately predicting pickup and drop-off counts, it's essential
to train a variety of models. This process includes iteratively adding and removing features and
evaluating model performance using the Test MSE (Mean Squared Error). The objective is to select the
model with the lowest Test MSE score, indicating the best predictive performance.

# Interaction Testing
To determine the best feature combination, the team investigated the presence of non-additivity in the
feature set. Specifically, the interaction between 'tempmax' and 'humidity' was examined to see how
these variables interact while influencing the dependent variables. The dataset was divided into two
halves based on 'tempmax' values, with one half containing observations with 'tempmax' above 70°F
and the other below. The interaction between ‘tempmax’ and ‘humidity’ was analyzed using the
following steps:

a. A new discrete variable, ‘tempmax70’, was introduced to categorize observations based on
whether ‘tempmax’ was above or below 70°F.
b. The dataset was segregated into two sets: one with ‘tempmax70’ above 70°F and the other with
‘tempmax70’ below 70°F.
c. For the dataset with ‘tempmax’ above 70°F, a multi-linear regression model was built, resulting
in a beta coefficient of 0.2058 for Humidity.
d. Similarly, for the dataset with ‘tempmax’ below 70°F, a multi-linear regression model was built,
with a beta coefficient of -0.1064 for Humidity.
e. Finally, the beta coefficient for ‘humidity’ in a multi-linear regression model built on the entire
dataset without segregation was noted as -0.0804.
f. The key observation was that the relationship between pickup count and ‘humidity’ changes
significantly when the outside temperature is below or above 70°F, suggesting a potential
interaction between ‘tempmax’ and ‘humidity’.
g. Subsequently, a multiple linear regression (MLR) model was constructed to incorporate the
interaction between ‘tempmax’ and ‘humidity’. This allowed for an examination of the additional
flexibility introduced to the model and whether there was an increase in the test mean squared
error (MSE) score. An increase in the residual errors and Test MSE was expected.

The above approach ensures that the bias-variance tradeoff is considered while tuning the models to
achieve the best prediction. To accomplish this, we used the itertools package to iteratively create a
series of models. Subsequently, we developed a custom function to generate the Training MSE, Test
MSE, the set of features used in training, and the total number of features utilized. This data enables us
to plot the Train/Test MSE against Model Flexibility, aiding in the selection of the optimal prediction
model. Tables 2.1 and 2.2 document the output of the itertools code execution and the custom function.

# Performance Evaluation
Upon reviewing the data in Tables 2.1 and 2.2, we identified model with index 561 for predicting
‘pickups’ and ‘dropoffs’ as they exhibited among the lowest Test MSE scores while maintaining a
balance between bias and flexibility. However, these features were not dropped when training a Linear
Regression model with Cross Validation, because we employed the power of regularization techniques
to enhance the reliability of the prediction. Details of the Beta coefficients and out-of-sample
performance scores can be found in Figures 2.1 and 2.2.

Subsequently, the team constructed a Ridge Regression model, which includes a regularization term
penalizing the size of coefficients. This model is particularly beneficial for reducing overfitting in the
presence of many correlated features. Ridge Regression strikes a balance between bias and variance
by penalizing large coefficients, making it suitable for datasets exhibiting multicollinearity. A comparison
of the performance between Linear Regression and Ridge Regression models is illustrated in Figure
2.3, highlighting the advantages of Ridge Regression in such scenarios.

In addition to Ridge Regression, the team sought to assess whether the features selected for the Linear
Regression and Ridge Regression models were optimal, or if there was an opportunity to further refine
the model by shrinking some less important features to zero. Therefore, a Lasso model was
constructed, which is simpler than Ridge Regression and offers even greater ease of interpretation.
Given that Lasso has been shown to improve prediction accuracy in some cases compared to Ridge
Regression, this approach was considered beneficial. Figure 2.3 provides an overview of the accuracy
scores achieved by the Lasso model we trained.

Lastly, we trained a KNN and ElasticNet. ElasticNet model combines the strength of Ridge and Lasso.
Since ElasticNet includes both L1 (Lasso) and L2 (Ridge) regularization terms in its loss function, it
allows the model to benefit from Lasso's ability to perform feature selection and Ridge's ability to handle
multicollinearity. Results of out of sample performance is presented in Figure 2.3.

# Predictive Analytics and Decision Startegy
A machine learning model developed by GWSB serves as a tool for Capital Bikeshare to make
informed business decisions to optimize their outcomes. To evaluate the quality of these decisions, the
project team recommends using Penalty-based and Quality of Service (QoS) metrics.

Penalty-based approach: This strategy aims to identify a business decision that minimizes the cost of
penalties. Using predictions from the model, Capital Bikeshare can deploy an optimal number of bikes
and docks to minimize their penalty costs. The following equations are integrated into GWSB's
prescriptive analytics code to optimize the business decision.

<table>
<tr>
<td style="border:1px solid black; padding:8px;">
<h4>Total Penalty = Alpha * Max (0, P<sub>b</sub> – D<sub>b</sub>) + Beta * Max (0, P<sub>d</sub> – D<sub>d</sub>) </h4>
</td>
</tr>
</table>…Equation (1)


This is the penalty ($) that Capital Bikeshare may incur if they fail to meet the demand predicted by
GWSB's ML model.

Alpha = Penalty ($) levied on Capital Bikeshare for every instance of unfulfilled bike demand. Project
team has agreed on $7.

Beta = Penalty levied on Capital Bikeshare for every instance of unfulfilled dock demand. Project team
has agreed on $2.

P<sub>b</sub> = No. of bikes predicted by model (Y_Pred_bikes)

D<sub>b</sub> = No. of bikes Capital Bikeshare chose to deploy based on model prediction (Biz_Decision_Bikes)

P<sub>d</sub> = No. of docks predicted by model (Y_Pred_docks)

D<sub>d</sub> = No. of docks Capital Bikeshare chose to deploy based on model prediction (Biz_Decision_Docks)

<table>
<tr>
<td style="border:1px solid black; padding:8px;">
<h4>Actual Penalty = Alpha * Max (0, A<sub>b</sub> – D<sub>b</sub>) + Beta * Max (0, A<sub>d</sub> – D<sub>d</sub>)</h4>
</td>
</tr>
</table>…Equation (2)


This is the penalty ($) that Capital Bikeshare may incur if their decision-making process is not informed
by the state-of-the-art machine learning model developed by GWSB.

A<sub>b</sub> = No. of bikes actually deployed by Capital Bikeshare (Y_Test_bikes) in the absence of GWSB’s
Machine Learning model

D<sub>b</sub> = No. of bikes Capital Bikeshare chose to deploy based on model prediction (Biz_Decision_Bikes)

A<sub>d</sub> = No. of docks actually deployed by Capital Bikeshare (Y_Test_docks) in the absence of GWSB’s
Machine Learning model

D<sub>d</sub> = No. of docks Capital Bikeshare chose to deploy based on model prediction (Biz_Decision_Docks)
D<sub>b</sub> and D<sub>d</sub> are the hypothetical benchmarks (which has been conveniently set to be equal to the business decision in case Capital Bikeshare adopts GWBS ML model) to evaluate the penalty that the
company may incur if they chose not to implement GWSB ML model.

To engineer the decision strategy, the project team built a custom function that -

a. Intakes the model prediction for each one of the X-Test observations (for both pickups and drop-
offs), and loops through 18 unique combinations of Pickup and Drop-off allocations ranging from
(0,17), (1,16), (2,15), (3,14) ‚ up to (17,0). For each one these probable decisions, the numbers
were fed to the equation (1) and the resulting cost was captured in a decision matrix (refer to
Table 3.1 in Appendix).

b. Next, the function will skim through the resulting Total Penalty and filter out the record with the
least cost. For this record, the code will retrieve the business decision (Db and Dd) and write to
a separate decision matrix (Table 3.2 in Appendix). The retrieved business decision is the
optimum one for the prediction made by ML model for the given X-Test record.

c. (a) and (b) will be repeated for all the remaining X-Test records and appended to Table 3.2.

d. Once we have the Least Total Penalty of all X-Test records, we calculate the Actual Penalty
using equation (2) wherein we feed business decision retrieved from each one of the least costs
and compare it with the number of bikes and docks deployed in reality (Y-Test) which was a
decision taken without being informed by the ML model delivered by GWSB. The Actual Penalty
is further written to Table 3.2.

e. In the subsequent step, we evaluate if the Total Penalty (which is a function of the model
prediction and a model-informed business decision) is less than the Actual Penalty. If yes, the
record is flagged as 1, else, 0. Refer to Table 3.2

f. Once we have the flag generated for all records, we calculate the probability of 1's which is
nothing but an indicator of the ML model performance. This probability along with the model’s
name is written to new table (refer Table 3.3). A probability of more than 0.5 indicates that on
average GWSB's ML model will enable Capital Bikeshare to reduce the penalties incurred from
unfulfilled demand.

g. Repeat (a) to (f) for all other models being trained to solve Capital Bikeshares business
conundrum.

<h4>Quality of Service (QoS) approach:</h4> This approach evaluates the efficacy of Capital Bikeshare service
maximizing the score calculated by the following formula.


<table>
<tr>
<td style="border:1px solid black; padding:8px;">
<h4>QoS<sub>m</sub> = Alpha * Min (P<sub>b</sub>, D<sub>b</sub>)/ P<sub>b</sub> + Beta * Min (P<sub>d</sub>, D<sub>d</sub>)/ P<sub>d</sub></h4>
</td>
</tr>
</table>…Equation (3)


QoS<sub>m</sub> = Quality of Service calculated by making a business decision that is informed by ML
model prediction

Alpha = Reward for every instance of fulfilled bike demand. Project team has agreed on a score
of 7.

Beta = Reward for every instance of fulfilled dock demand. Project team has agreed on a score
of 2.

<table>
<tr>
<td style="border:1px solid black; padding:8px;">
<h4>QoS<sub>wm</sub> = Alpha * Min (A<sub>b</sub>, D<sub>b</sub>)/ P<sub>b</sub> + Beta * Min (A<sub>d</sub>, D<sub>d</sub>)/ P<sub>d</sub> </h4>
</td>
</tr>
</table>…Equation (4)

QoS<sub>wm</sub> = Quality of Service calculated by making a business decision that is not informed by ML
model prediction

Every unit of demand fulfilled by the company is rewarded by a certain number of points denoted by
Alpha and Beta. Similar to the Penalty approach, the team has designed a custom function which
executes the following steps.

a. Intakes the model prediction for each one of the X-Test observations (for both pickups and drop-
offs), and loops through 18 unique combinations of Pickup and Drop-off allocations ranging from
(0,17), (1,16), (2,15), (3,14) ‚ up to (17,0). For each one these probable decisions, the numbers
were fed to the equation (3) and the resulting cost was captured in a decision matrix (refer to
Table 3.4 in Appendix).

b. Next, the function will skim through the resulting Quality of Service and filter out the record with
the highest QoSm. For this record, the code will retrieve the business decision (Db and Dd) and
write to a separate decision matrix (Table 3.5 in Appendix). The retrieved business decision is
the optimum one for the prediction made by ML model for the given X-Test record.

c. (a) and (b) will be repeated for all the remaining X-Test records and appended to Table 3.5.

d. Once we have the best QoSm for all X-Test records, we calculate the QoSwm using equation (4)
wherein we feed business decision retrieved from each one of the best QoSm and compare it
with the number of bikes and docks deployed in reality (Y -Test) which was a decision taken
without being informed by the ML model delivered by GWSB. The QoSwm is further written to
Table 3.5.

e. In the subsequent step, we evaluate if the QoSm (which is a function of the model prediction and
a model-informed business decision) is greater than QoSwm. If yes, the record is flagged as 1,
else, 0. Refer to Table 3.5

f. Once we have the flag generated for all records, we calculate the probability of 1's which is
nothing but an indicator of the ML model performance. This probability along with the model's
name is written to new table (refer Table 3.6). A probability of more than 0.5 indicates that on
average, GWSB's ML model will enable Capital Bikeshare to maximize their Quality of Service.

g. Repeat (a) to (f) for all other models being trained to solve Capital Bikeshares business
conundrum.

# Data-driven recommendation for Capital Bikeshare
If Capital Bikeshare aims to minimize penalties incurred from failing to meet demand, they stand to
benefit from implementing the <b>ElasticNet model</b> developed by GWSB. Conversely, if their goal is to
improve the Quality of Service, the <b>Lasso</b> model would be more suitable. These decisions were
informed by comparing the probabilities outlined in Tables 3.3 and 3.6. The models with the highest probability values were selected as the preferred choices. Regarding the <b>Penalty strategy</b>, the
<b>ElasticNet model</b> yielded a probability of <b>0.62</b>, indicating that on average, the GWSB model would
reduce penalty costs if deployed in a production environment. The QoS strategy on the other hand is
interested to see an above average probability of maximizing the reward emerging from those
instances where the company has managed to fulfill the demand. For the <b>QoS strategy</b>, the goal is to
maximize rewards from instances where demand is fulfilled, aiming for an above-average probability. A
quick look at Table 3.6 suggests that the <b>Lasso</b> model would enable Capital Bikeshare to maximize
rewards, as it has the highest probability of <b>0.62</b> among the models considered. This indicates that in
more than 6 out of 10 instances, Capital Bikeshare could enhance the quality of their service by
deploying the GWSB model in production.

# Appendix

<h3>Table 1.1: Mutual Information Scores for Pickup features</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/098ea154bd6b72ffb9a8e59457fbb2e0704eee0d/Auxiliary/Mutual%20Information%20Scores%20for%20Pickup%20features.png" width="800" />

<h3>Table 1.2: Mutual Information Scores for Drop Off features</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/098ea154bd6b72ffb9a8e59457fbb2e0704eee0d/Auxiliary/Mutual%20Information%20Scores%20for%20Drop%20Off%20features.png" width="800" />

<h3>Table 2.1: MSE vs Model Flexibility – Pickups</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/098ea154bd6b72ffb9a8e59457fbb2e0704eee0d/Auxiliary/MSE%20vs%20Model%20Flexibility%20%E2%80%93%20Pickups.png" width="800" />

<h3>Table 2.2: MSE vs Model Flexibility – Drop Offs</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/098ea154bd6b72ffb9a8e59457fbb2e0704eee0d/Auxiliary/MSE%20vs%20Model%20Flexibility%20%E2%80%93%20Drop%20Offs.png" width="800" />

<h3>Table 3.1: Business Decision Matrix (level 1) – Penalty Strategy</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Business%20Decision%20Matrix%20(level%201)%20%E2%80%93%20Penalty%20Strategy.png" width="800" />

<h3>Table 3.2: Business Decision Matrix (level 2) – Penalty Strategy</h3>

<h3>Linear Regression – head:</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Linear%20Regression%20%E2%80%93%20head.png" width="800" />

<h3>Ridge Regression - head:</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Ridge%20Regression%20-%20head.png" width="800" />

<h3>Lasso – head:</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Lasso%20%E2%80%93%20head.png" width="800" />

<h3>ElasticNet – head:</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/ElasticNet%20%E2%80%93%20head.png" width="800" />

<h3>KNN – head:</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/KNN%20%E2%80%93%20head.png" width="800" />

<h3>Table 3.3: Business Decision Matrix (Probabilities) – Penalty Strategy</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Business%20Decision%20Matrix%20(Probabilities)%20%E2%80%93%20Penalty%20Strategy.png" width="800" />

<h3>Table 3.4: Business Decision Matrix (level 1) – QoS Strategy</h3>

<img src="https://github.com/arnab-raychaudhari/Pedal-Perfect/blob/1d6feab41eb1a5c9d3e6da8639dbfed15eae8218/Auxiliary/Business%20Decision%20Matrix%20(level%201)%20%E2%80%93%20QoS%20Strategy.png" width="800" />

