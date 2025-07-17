# FB Prophet Demand Forecasting & Visualizations of Store Sales Data

A comprehensive time series forecasting project using Facebook Prophet to predict store sales data from Favorita stores in Ecuador. This project includes data preprocessing, exploratory analysis, hyperparameter tuning, and model evaluation with cross-validation.

## 📊 Project Overview

This project demonstrates end-to-end time series forecasting using FB Prophet on retail sales data. The implementation includes:

- **Data preprocessing and aggregation** of store sales by product family
- **Exploratory data analysis** with volume-based category grouping
- **Prophet model configuration** with seasonality and holiday effects
- **Hyperparameter tuning** using cross-validation
- **Model evaluation** with MAPE and RMSE metrics
- **Forecasting visualization** and performance analysis

## 🗂️ Dataset

The dataset contains sales data from Favorita stores in Ecuador, sourced from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=train.csv).

### Data Structure
- **Date Range**: January 1, 2013 - August 31, 2017
- **Training Period**: August 15, 2015 - June 30, 2017
- **Forecast Period**: July 1, 2017 - July 30, 2017 (30 days)

### Features
- `id`: Unique identifier for each record
- `date`: Sales record date
- `store_nbr`: Store identifier
- `family`: Product category (AUTOMOTIVE, BABY CARE, BEAUTY, BEVERAGES, etc.)
- `sales`: Sales amount for the product category
- `onpromotion`: Promotion indicator (0 = no promotion)

## 🛠️ Installation

```bash
# Required packages
pip install numpy pandas pandasql matplotlib seaborn
pip install python-dateutil holidays
pip install prophet
```

## 🚀 Usage

### 1. Data Preprocessing

```python
# Load and preprocess data
df = pd.read_csv('store_data.csv')
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

# Aggregate by date and product family
agg_df = df.groupby(['date','family']).agg({'sales':'sum'}).reset_index()
total_sales_df = agg_df.pivot(index='date', columns='family', values='sales')
```

### 2. Volume-Based Category Analysis

The project categorizes product families into three volume tiers:
- **Low Volume**: Bottom 33rd percentile of average daily sales
- **Mid Volume**: 33rd to 66th percentile
- **High Volume**: Top 33rd percentile (e.g., PRODUCE, GROCERY categories)

### 3. Prophet Model Configuration

```python
# Basic model setup
m = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    holidays=holiday  # Ecuador holidays included
)

# Fit model and generate predictions
m.fit(train_set)
future = m.make_future_dataframe(periods=prediction_days)
forecast = m.predict(future)
```

### 4. Hyperparameter Tuning

The project implements grid search for optimal parameters:

```python
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}
```

**Best Parameters Found:**
- `changepoint_prior_scale`: 0.1
- `seasonality_prior_scale`: 1.0
- **RMSE**: 17,703.48

### 5. Model Evaluation

The project uses multiple evaluation metrics:
- **MAPE (Mean Absolute Percentage Error)**: 4.88%
- **RMSE (Root Mean Square Error)**: 17,703.48
- **Cross-validation**: 365-day initial training, 30-day periods, 30-day horizon

## 📈 Key Features

### Holiday Integration
- Incorporates Ecuador public holidays with custom windows
- Holiday effects: -2 days before to +1 day after each holiday

### Seasonality Modeling
- **Yearly seasonality**: Captures annual patterns
- **Weekly seasonality**: Handles day-of-week effects
- **Multiplicative seasonality**: Better fits retail sales patterns

### Cross-Validation Framework
- Time series cross-validation with rolling windows
- Systematic hyperparameter optimization
- Robust performance evaluation across multiple time periods

## 📊 Results

### Model Performance
- **MAPE**: 4.88% (excellent accuracy for retail forecasting)
- **RMSE**: 17,703.48
- **Forecast Horizon**: 30 days ahead

### Category Insights
- High-volume categories (PRODUCE, GROCERY) show strong seasonal patterns
- Model performs best on categories with consistent demand patterns
- Holiday effects significantly impact forecasting accuracy

## 🔧 Functions

### `missing_data(input_data)`
Returns DataFrame with null percentage and data types for each column.

### `mape(actual, pred)`
Calculates Mean Absolute Percentage Error between actual and predicted values.

## 📁 File Structure

```
├── store_data.csv          # Raw sales data
├── prophet_forecasting.py  # Main analysis script
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## 🎯 Future Enhancements

- **Multi-step forecasting**: Extend prediction horizon
- **Ensemble methods**: Combine Prophet with other models
- **External regressors**: Add weather, economic indicators
- **Automated retraining**: Implement model refresh pipeline
- **Real-time deployment**: API for live forecasting

## 📚 References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [Kaggle Store Sales Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/)
- [Ecuador Holidays Package](https://pypi.org/project/holidays/)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements. Areas for contribution:
- Additional evaluation metrics
- Advanced visualization techniques
- Model interpretability features
- Performance optimization

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project demonstrates time series forecasting techniques and should be adapted based on specific business requirements and data characteristics.
