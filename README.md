# Nigeria-Housing-Price-Prediction

## Overview of project and Hackathon

Wazobia Real Estate Limited is a prominent real estate company operating in Nigeria. With a vast portfolio of properties, they strive to provide accurate and competitive pricing for houses. However, they have been facing challenges in accurately predicting the prices of houses in the current market. To overcome this hurdle, Wazobia Real Estate Limited is seeking the expertise of data scientists like you to develop a robust predictive model.

The objective of this hackathon is to create a powerful and accurate predictive model that can estimate the prices of houses in Nigeria. By leveraging the provided dataset, you will analyze various factors that impact house prices, identify meaningful patterns, and build a model that can generate reliable price predictions. The ultimate goal is to provide Wazobia Real Estate Limited with an effective tool to make informed pricing decisions and enhance their competitiveness in the market.

By participating in this hackathon, you have the opportunity to make a significant impact on the operations and growth of Wazobia Real Estate Limited. Your data-driven solution will empower the company to overcome their pricing challenges, improve their market position, and deliver enhanced value to their customers.

The hackathon was organised by Data Scientists Network, formerly known as Data Science Nigeria (DSN) (datasciencenigeria.org)

[PROJECT DATA LINK](https://zindi.africa/competitions/free-ai-classes-in-every-city-hackathon-2023/data)

### Explore and prepare the data
Exploration of the data was done via *ZINDI DSN HOUSING PROJECT- Per Project.ipynb* jupyter notebook file
- Explore data
  - Checked the Data Structure and columns
  - Checked the numbers of features and observations in the data
  - Checked the inconsistency in column names and corrected.
- prepare data
  - Checked the distribution of the data
  - Checked for missing values in the numerical features (filled with median)
  - Checked for outliers and removed
  - Checked for Duplicates
- Feature Engineering
    - Created additional features to improve model predictive power
- train data
  - Catergorical variables were encoded using the DictVectorizer library.
  - trained 4 models and the best model (Xgboost) was ascertained it to produce the best model with hyper parameters

#### Get copies of the project and dependencies, you can clone the repo.

```
git clone https://github.com/kabiromohd/Nigeria-Housing-Price-Prediction.git
```
    
### Model deployment to web services
- Dash was used for web deployment via *HOUSING_PROJECT.py* script.

### Dependencies 
- They are listed in the *requirement.txt* file in the git repo and used for cloud deployment on [Render](render.com)

### Deployment to Cloud via render
![Render deply](https://github.com/kabiromohd/Nigeria-Housing-Price-Prediction/assets/121871052/c6c2edf4-ec28-42f6-91a7-5138ef7794fc)

### Live Deployment Link
[Project Link](https://housing-price-project.onrender.com)
