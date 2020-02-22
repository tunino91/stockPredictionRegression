# Linear Regression Stock Prediction

## Used Features
'Adj. Close','HL_PCT','PCT_change','Adj. Volume
Where
HL_PCT:High to low percentage change (High - Low) / Low
PCT_change:daily percentage change (Close - Open) / Open

## Cleaning
All the NaNs are filled with -9999 to count as an outlier by the sklearn LinearRegression object's "fit" function.

## Adjustable Parameter
You can determine how much of historical data you want to incorparate in your predictions by changing the **forecast_out** variable
