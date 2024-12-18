# Stock Option Pricing in Financial Markets Using Machine Learning


## Table of Contents

1. [Abstract](#abstract)
2. [Quantitative Analysis Method](#quantitative-analysis-method) 
3. [Dataset Description](#dataset-description)
4. [Machine Learning Algorithms Used](#machine-learning-algorithms-used)
5. [Training Results for Optimal Model Selection](#training-results-for-optimal-model-selection)
6. [Pricing Result Visualisation](#pricing-result-visualisation)
7. [Prediction Error Visualisation](#prediction-error-visualisation)
8. [Feature Importance of the Model](#feature-importance-of-the-model)
9. [Future Improvements](#future-improvements)

<br>

## Abstract
This financial engineering project applies machine learning to price stock option using a large dataset (74,492 rows and 21 columns) from Yahoo Finance, covering 50 companies across various sectors. Among five regression models, Random Forest demonstrated the best predictive performance with low error rates and was selected for the final prediction. The model effectively identified key factors influencing option pricing, such as implied volatility, strike price, and time to maturity. It achieved strong evaluation metrics on the test set, with predictions closely aligning to actual pricing values.

<br>

## Quantitative Analysis Method
The analysis was performed using Python in Google Colab. The methodology involved several key steps:

1. **Data Collection**:
   - Datasets were sourced from Yahoo Finance for 50 companies across various sectors.

2. **Exploratory Data Analysis (EDA)**:
   - Statistical methods like summary statistics and histograms were used to explore the data.

3. **Data Inspection and Cleaning**:
   - Missing and empty rows were removed to ensure data integrity.
     
4. **Feature Engineering**:
   - Additional features (time_to_maturity, moneyness, and bid_ask_spread) were derived to enhance model context.

5. **Feature Selection**:
   - Relevant features were selected based on correlation and Variance Inflation Factor (VIF) analysis.
     
6. **Machine Learning Model Training**:
   - Five regression models (Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and Support Vector Regression) were trained, with hyperparameter tuning using RandomizedSearchCV. The best-performing model was selected for further analysis.

7. **Model Evaluation and Pricing Visualisation**:
   - The final model was evaluated using regression metrics (MSE, RMSE, MAE, MAPE, R-squared, and Explained Variance) with visualizations such as scatter plots and histograms created to analyse prediction accuracy.

<br>

## Dataset Description
This project utilises the latest stock options datasets from 50 companies across various sectors sourced from Yahoo Finance, with datasets that refresh upon each fetch. The companies include major names such as Apple, Microsoft, Tesla, NVIDIA, and more. The raw dataset comprises 79,853 rows and 18 columns, where each row represents an options contract and columns detail parameters like strike price, implied volatility, expiration date, and more. After cleaning, 74,492 rows remain, with missing values removed. Additionally, three derived parameters: time to maturity, moneyness, and bid-ask spread were added to enhance the dataset’s analytical depth.


| Column Name         | Data Type                | Description                         | Count   | Mean         | Min           | Max           | Std Dev       |
|---------------------|--------------------------|-------------------------------------|---------|--------------|---------------|---------------|---------------|
| contractSymbol      | object                  | Unique identifier for each contract | 74,492  | N/A          | N/A           | N/A           | N/A           |
| lastTradeDate       | datetime64[ns, UTC]     | Last trading date                   | 74,492  | 2024-10-31   | 2016-03-11    | 2024-11-27    | N/A           |
| strike              | float64                 | Strike price                        | 74,492  | 470.57       | 0.50          | 7,500.00      | 719.82        |
| lastPrice           | float64                 | Last traded price                   | 74,492  | 74.02        | 0.01          | 3,646.27      | 164.58        |
| bid                 | float64                 | Current bid price                   | 74,492  | 33.91        | 0.00          | 3,208.00      | 144.38        |
| ask                 | float64                 | Current ask price                   | 74,492  | 34.89        | 0.00          | 3,232.00      | 146.93        |
| change              | float64                 | Price change                        | 74,492  | -0.23        | -367.47       | 774.66        | 7.82          |
| percentChange       | float64                 | Percent price change                | 74,492  | -1.41        | -99.93        | 7,400.00      | 62.01         |
| volume              | float64                 | Trading volume                      | 74,492  | 145.81       | 0.00          | 228,429.00    | 2,062.93      |
| openInterest        | float64                 | Open contracts                      | 74,492  | 30.11        | 0.00          | 41,975.00     | 388.52        |
| impliedVolatility   | float64                 | Implied volatility of options       | 74,492  | N/A          | N/A           | N/A           | N/A           |
| inTheMoney          | object                  | Indicates if option is in the money | 74,492  | N/A          | N/A           | N/A           | N/A           |
| contractSize        | object                  | Size of the option contract         | 74,492  | N/A          | N/A           | N/A           | N/A           |
| currency            | object                  | Currency of the option              | 74,492  | N/A          | N/A           | N/A           | N/A           |
| option_type         | object                  | Type (call/put)                     | 74,492  | N/A          | N/A           | N/A           | N/A           |
| expiration_date     | datetime64[ns]          | Option expiration date              | 74,492  | 2025-06-30   | 2024-11-29    | 2027-01-15    | N/A           |
| stock               | object                  | Stock ticker                        | 74,492  | N/A          | N/A           | N/A           | N/A           |
| stock_price         | float64                 | Current stock price                 | 74,492  | 459.39       | 17.18         | 5,223.15      | 851.55        |
| time_to_maturity    | float64                 | Time to expiration in years         | 74,492  | 0.58         | -0.0027       | 2.13          | 0.63          |
| moneyness           | float64                 | Measure of intrinsic value          | 74,492  | -0.52        | -22.19        | 0.99          | 2.39          |
| bid_ask_spread      | float64                 | Difference between bid and ask      | 74,492  | 0.98         | -94.00        | 40.00         | 3.21          |

**Table 1: Summary of Dataset Columns and Key Statistics**

<br>

## Machine Learning Algorithms Used
- **Linear Regression**: Simple, efficient model for linear relationships.
- **Ridge Regression**: Regularized model to prevent overfitting. 
- **Random Forests**: Ensemble model for large, non-linear data.
- **Gradient Boosting**: Reduces bias and variance via weak learner combinations.
- **Support Vector Regressio**: Excels in high-dimensional spaces.

<br>

## Training Results for Optimal Model Selection

| Model                     | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) | Mean Absolute Percentage Error (MAPE) | R-squared (R²) | Explained Variance Score |
|----------------------------|---------------------------|--------------------------------|--------------------------------|-----------------------------------------|----------------|-------------------------|
| Linear Regression           | 10525.431807               | 102.593527                       | 48.816828                        | 10853.778479                               | 0.607325       | 0.607326                 |
| Ridge Regression            | 10525.430080               | 102.593519                       | 48.816829                        | 10854.118772                               | 0.607325       | 0.607326                 |
| Random Forest               | 676.218183                 | 26.004195                        | 7.587412                         | 48.047761                                 | 0.974772       | 0.974775                 |
| Gradient Boosting           | 3508.271399                | 59.230663                        | 19.959248                        | 2619.825724                                | 0.869116       | 0.869122                 |
| Support Vector Regression (SVR) | 15093.463847          | 122.855459                       | 45.219775                        | 5738.897342                                | 0.436904       | 0.472096                 |

**Table 2: Model Evaluation Metrics for Different Regression Models**

Random Forest significantly outperforms the other models, which demonstrates the best overall predictive accuracy with low error metrics (MSE: 676.22, RMSE: 26.00, MAE: 7.59) and a high R-squared value (0.97), indicating it captures the majority of the variance in the data. Gradient Boosting performs well (R-squared: 0.87), but still falls behind Random Forest in terms of prediction accuracy. Linear and Ridge Regression exhibit similar performance, with higher errors (MSE: 10525) and lower R-squared (0.61), indicating limited model fit. Support Vector Regression (SVR) performs the worst, with the highest MSE, RMSE, and MAPE, and the lowest R-squared (0.44), suggesting it is not suitable for this dataset. 

Hence, Random Forest was selected for hyperparameter tuning and option pricing since it can better provide valuable insights into market behavior by identifying key parameters that influence price movements, such as implied volatility, strike price, and time to maturity.

<br>

## Pricing Result Visualisation

<div style="display: flex; justify-content: center; align-items: flex-start;">
    <figure style="text-align: center; margin: 0;">
        <img src="/images/Actual vs Predicted Prices.png" alt="Scatter plot of actual vs. predicted option prices for the Random Forest model" style="width: auto; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
    </figure>
</div>

**Figure 1: Scatter plot of actual vs. predicted option prices for the Random Forest model**

The red dashed line in the scatter plot represents the ideal fit, and the data points are closely aligned with the diagonal, indicating strong predictive performance and minimal prediction error. This alignment demonstrates the model's ability to accurately capture the underlying relationships in the data and generalize effectively for option pricing.

<br>

## Prediction Error Visualisation

<div style="display: flex; justify-content: center; align-items: flex-start;">
    <figure style="text-align: center; margin: 0;">
        <img src="/images/Histogram of Prediciton Errors.png" alt="Histogram of prediction errors (Actual - Predicted) for the Random Forest model" style="width: auto; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
    </figure>
</div>

**Figure 2: Histogram of prediction errors (Actual - Predicted) for the Random Forest model**

The prediction error histogram displayed a distribution closely resembling a normal curve with values clustered around zero and a relatively small spread. This indicates that the model's predictions are consistently accurate with minimal fluctuation in the error values.

<br>

## Feature Importance of the Model

<div style="display: flex; justify-content: center; align-items: flex-start;">
    <figure style="text-align: center; margin: 0;">
        <img src="/images/Feature Importance of Model.png" alt="Feature importance of the Random Forest model" style="width: auto; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
    </figure>
</div>

**Figure 3: Feature importance of the Random Forest model**

The analysis shows that the bid price is the most important factor with a score close to 0.6, followed by stock price at 0.17 and moneyness at 0.15. Implied volatility scores lower at 0.07, while time to maturity and bid-ask spread have minimal impact. This highlights the model’s focus on key variables like bid price, stock price, and moneyness.

The bid price reflects market demand and significantly influences the option's value. Stock price determines the option's intrinsic value. Moneyness indicates whether the option is in, at, or out of the money. These three parameters directly influence an option's value as they focus on immediate and market-driven factors by reflecting market demand, the asset's current price, and the option’s intrinsic value. Although implied volatility affects the option’s premium, its influence is smaller compared to the other factors. Time to maturity and bid-ask spread have minimal impact because they are less correlated with immediate option value, which is driven by market-sensitive factors like bid price, stock price, and moneyness.

<br>

## Future Improvements
Future work could focus on integrating real-time data for live pricing and using more advanced models to enhance accuracy and scalability. Leveraging GPUs would speed up training and enable the switch from randomized search to grid search for better optimization. Migrating the Jupyter notebook to a more scalable environment would improve efficiency and simplify deployment.

