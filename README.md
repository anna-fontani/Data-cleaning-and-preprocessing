# Data Cleaning, Preprocessing and EDA

This example project demonstrates data cleaning and preprocessing before ML models application.  
It also includes data melting as initial dataset lacked essential information (country area) necessary for analysis.

EDA (Exploratory Data Analysis) is performed to study data in-depth. 

## Research question
- Explore and analyze factors influencing farms performance in European Union in general and Ireland in particular.

## Data sources and licenses
In this Notebook we will use data from .csv and .xlsx files available in EuroStat and FAO:

1.European Agriculture Farm Structure Indicators 2016.

2.Country Area

Copyright: Eurostat, the Statistical Office of the European Union.

License: Datasets are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Reuse is allowed, provided appropriate credit is given and changes are indicated.

## EDA
Preprocessing was necessary as the dataset contained information irrelevant to research and missing data that had to be imputed. It also lacked country size and required metric change. 
European farm structure dataset contains general agricultural information on 27 EU countries. Dataset describes country size, used agricultural area, financial output and labour-related statistics. Dataframe contains string country names and continuous variables with different measurements.

### Missing Values Imputation

Missing values had to be addressed since a number of ML models do not accept missing values. According to Kang (2013), “Missing data can reduce the statistical power of a study and can produce biased estimates, leading to invalid conclusions”. Missing values usually have a complex statistical relationship. Common approaches to missing values imputation include replacing them with mean, median, or interpolated values depending on data specifics. In our case, features containing missing values have left skewness: standard_output_mil, nonfamily_labour_ths.

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/c9cfd6a5-0023-4d0c-89b9-a6c458830e53)
![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/ab9af740-4c34-41c6-ba5f-999c08abc14e)

Figure 1. Total labour persons (left) and standard financial output (right) histogram plots.

Czechia lacked financial output, and this country is particularly interesting to compare with Ireland due to similar size. 

Skewed data shouldn’t be imputed using mean values. Mean value substitution generally leads to biased results especially in skewed datasets (Kang, 2013). Deleting missing values in relatively small datasets may lead to biased results and loss of essential observations. Hence, imputation method IterativeImputer from sklearn library was used, which by default is based on BayesianRidge regression (Towards Data Science, 2019). 

Resulting feature characteristics are demonstrated by Figure 2.

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/7d5d3d14-7b99-4d26-bfbb-0a98ae56c4f5)
![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/fd84eee2-0145-4772-bc5a-0fa6473522e0)

Figure 2. Total labour persons (left) and standard financial output (right) histogram plots after imputation.

### Descriptive statistics

Table 1. Descriptive statistics of European farm structure dataframe. 

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/dcc2fe68-ed80-4ec1-a2ab-a10948f71fac)

Following pair plot reflects data relationships.

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/3036bd12-2f9a-468d-9fe0-825adbdc9ad0)

Figure 3. Pairwise plot of European farms structure dataset

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/9af4ca57-4f38-4ae0-8f3c-0fc12d43afdd)

Figure 4. Density plots for main features.

Pairwise plot is useful as it demonstrates relationships between features with diagonal plots being the univariate histograms or distributions for each feature (Sarath, 2019). Diagonal plots show clear left skewness. Some plots have points forming a line – a higher correlation of farms number and total labour, country area and used agricultural area. Other features demonstrate more scattered picture with different outliers and underlying factors. Interestingly, financial output plot shows that countries with small area and relatively low number of workforce can reach great profits in agriculture. 

To deepen understanding of data, we added calculated values: output per farm, labour per farm, area per farm, ratio of fully trained managers. 

Table 2. List of countries sorted by labour per farm, high to low. 

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/7d228521-4369-4963-a8d1-833860c4edc0)

Several interactive geographic graphs were added to demonstrate general picture of financial output per farm and labour per farm. 

Table 3. List of countries sorted by output per farm (in millions of euro), high to low. 

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/43d83fc3-110f-41ef-a6df-fafcdec8e08c)

Small countries have a potential of optimization to reach high results demonstrated by top four countries: Netherlands, Denmark, Belgium, and Luxembourg.

### Insights

- Ireland ranks 13th in EU financial output per farm. When compared to other countries within the same country area group, Ireland shows 63% better results than Latvia, and 67% better than Lithuania. Austria shows 1% better result and Slovakia – 4% better result than Ireland.
- In terms of financial output per farm, significantly higher than Ireland (with smaller territory) results are demonstrated by Luxembourg, Belgium, Denmark and Netherlands.
- Ireland is 11th in area per farm metric and 12th in farms number. Considering relatively small size of Ireland, the number of farms is high. Only Lithuania has more farms in the same country size group.
- Ireland ranks 4th in managers with full training ratio too all labour, which demonstrates high standards in labour training. 

### Scaling 

Difference in magnitude in features can create problems for ML algorithms. Algorithms that use gradient descent require scaling due to possible step size differences. Distance-based algorithms like k-Means are significantly affected by range of features as they are based on distances between data points (Bhandari, 2024). Our dataframe contained data with different magnitude and required it to be addressed. 
When comparing normalization and standardization, standardization is less sensitive to outliers, preserves relationships between data, and can be used for data with unknown distribution (Bhandari, 2024). For these reasons, StandardScaler was applied to the dataframe.

Table 4. Scaling results.

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/a3cd53a3-4655-4f55-9cf3-1c2e6afc881f)

### Correlation and Heat Map

Correlation coefficient helps to see relationships between two variables. It demonstrates the measure of the strength of association between two variables. The value of correlation coefficient can be between -1 to +1, where 1 means that they are highly correlated and 0 means no linear relationship between values, the value r < 0 indicates negative correlation (Stojiljković, n.d.).

A correlation heatmap displays the correlation between multiple variables as a color-coded matrix. The color of each cell represents the strength and direction of the correlation, with darker colors indicating stronger correlations (Chip, 2023).

To perform linear regression in ML section, the numeric target variable should be linearly related to at least one another numeric variable. Correlation heatmap reveals some useful insights. 

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/14db82cd-58d0-401e-87fe-0c89d6e205bc)

Considering the financial output of farms throughout EU, correlation heatmap and matrix (cells 49-50) demonstrate high correlation of financial output and number of managers with basic training (correlation = 0.52), nonfamily labour employed (correlation = 0.97), country area (correlation = 0.87), and agricultural area used (correlation = 0.89). 

Whereas total area and agricultural area usage can be called self-evident factors, for the more land, the higher the profit can be, high correlation with nonfamily labour and number of managers with basic training lead to interesting insights. Notably, managers with full training or only practically trained have significantly lower correlation results. Total labour and farms number have very low correlation, which could signify importance of other factors when we discuss financial output, such as training, technology, government support, and others. 

In future research, it would be essential to consider areas that demand managers with different level of training. Nonfamily labour employed can also signify other developments of farms (technological, professional), which are not displayed by data available for this research. 

In next projects we will apply different ML models, evaluate their results, and perform hyperparameters tuning to optimize models performance. 

## References

Bhandari, A. (2024). Feature Scaling: Engineering, Normalization, and Standardization. [online] Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization.

Kang, H. (2013). The Prevention and Handling of the Missing Data. Korean Journal of Anesthesiology, [online] 64(5), pp.402–406. doi:https://doi.org/10.4097/kjae.2013.64.5.402.

Towards Data Science (2019). Preprocessing: Regression Imputation of Missing Continuous Values. [online] Towards Data Science. Available at: https://towardsdatascience.com/preprocessing-regression-imputation-of-missing-continuous-values-f612179bafb4.

Sarath, S.L. (2019). Pairplot Visualization. [online] Medium. Available at: https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6.

Stojiljković, M. (n.d.). NumPy, SciPy, and Pandas: Correlation With Python. [online] realpython.com. Available at: https://realpython.com/numpy-scipy-pandas-correlation-python [Accessed 5 Apr. 2024].

Chip (2023). How to read a correlation heatmap? [online] Quanthub. Available at: https://www.quanthub.com/how-to-read-a-correlation-heatmap/.
