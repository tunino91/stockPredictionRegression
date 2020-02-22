# Stock Prediction with Linear Regression

## Used Features
1) Adj. Close: Close price of the day
2) HL_PCT: High to low percentage change (High - Low) / Low
3) PCT_change: Daily percentage change (Close - Open) / Open
4) Adj. Volume: Daily traded volume

## Cleaning
All the NaNs are filled with -9999 to count as an outlier by the sklearn LinearRegression object's "fit" function.

## Adjustable Parameter
You can determine how much of historical data you want to incorparate in your predictions by changing the ***forecast_out*** variable
