# Load Necessary Libraries
library(quantmod)
library(readr)
library(pacman)
library(xts)
library(zoo)
library(astsa)
library(nnet)
library(stats)
library(RSNNS)
library(Rcpp)
library(TTR)
library(httr)
library(jsonlite)
library(lubridate)
library(utils)

# Data Retrieval
hour_req <- "https://marketdata.tradermade.com/api/v1/timeseries?currency=EURUSD&api_key=API_KEY&start_date=2024-05-08-08:00&end_date=2024-05-17-07:00&format=records&interval=hourly"

data_hour_raw <- GET(url = hour_req)
data_hour_text <- content(data_hour_raw, "text", encoding = "UTF-8")
data_hour_json <- fromJSON(data_hour_text, flatten=TRUE)
dataframe_hour <- as.data.frame(data_hour_json["quotes"])

col_order <- c("quotes.date", "quotes.open", "quotes.close",
               "quotes.high", "quotes.low")
eur_usd <- dataframe_hour[, col_order]

names(eur_usd)[1] <- "date"
names(eur_usd)[2] <- "open"
names(eur_usd)[3] <- "close"
names(eur_usd)[4] <- "high"
names(eur_usd)[5] <- "low"

eur_usd$date <- ymd_hms(eur_usd$date)

# Subset the data frame to include only the relevant columns
eur_usd <- eur_usd[, c("date", "open", "high", "low", "close")]

# Convert the subsetted data frame to an xts object
eur_usd_xts <- as.xts(eur_usd)

# Convert the open prices to a time series object
eur_usd_ts <- as.ts(eur_usd$open)

# Plot the EUR/USD price time series
plot(eur_usd_ts)

# Transform the EUR/USD price data by taking the natural logarithm
y <- ts(log(eur_usd_ts), frequency = 12)

# Perform time series decomposition (multiplicative) to separate components
decomp <- decompose(y, type = "multiplicative")
plot(decomp)
plot(decomp$trend, ylab = "Trend")

# Analyze the distribution of the residuals (random component)
random <- na.omit(decomp$random)
plot(density(random))
qqnorm(random, pch = 1, frame = TRUE)
qqline(random, col = "steelblue", lwd = 2)

# Compute the autocorrelation function (ACF) and partial autocorrelation function (PACF)
acf(y)
pacf(y)

# Data Normalization
range_data <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
eur_usd_norm <- range_data(eur_usd_ts)
min_data <- min(eur_usd_ts)
max_data <- max(eur_usd_ts)
eur_usd_zoo <- as.zoo(eur_usd_norm)

# Create Lagged Variables
lags <- lapply(1:12, function(k) Lag(eur_usd_zoo, k = k))
x <- do.call(cbind, c(lags, list(eur_usd_zoo)))
x <- x[-(1:12),]

# Neural Network Training
n <- nrow(x)
set.seed(2018)
n_train <- round(0.8 * n)
train <- sample(1:n, n_train, FALSE)

inputs <- x[, 2:13]
outputs <- x[, 1]

fit <- elman(inputs, outputs, size = c(ncol(inputs), 2), maxit = 1000)
plotIterativeError(fit)
plotRegressionError(outputs, fit$fitted.values)

# Unscale the actual output data
unscale_data <- function(x, max_x, min_x) {
  (x * (max_x - min_x) + min_x)
}

fitted_values_unscaled <- unscale_data(fit$fitted.values, max_data, min_data)
actual_values_unscaled <- unscale_data(outputs, max_data, min_data)

# Prepare input for next prediction
last_known <- tail(x, 1)
predictions <- numeric(10)

for (i in 1:10) {
  next_pred <- predict(fit, last_known[, -1])
  predictions[i] <- next_pred
  last_known <- cbind(last_known[, -1], next_pred)
}

# Unscale predictions
predictions_unscaled <- unscale_data(predictions, max_data, min_data)

# Plot the results
plot(as.ts(eur_usd_ts), type = "l", col = "blue", ylab = "Price", xlim = c(1, length(eur_usd_ts) + 10))
lines(length(eur_usd_ts) - length(fitted_values_unscaled) + 1:length(fitted_values_unscaled), fitted_values_unscaled, col = "green")
lines(length(eur_usd_ts):(length(eur_usd_ts) + 9), predictions_unscaled, col = "red")
legend("topleft", legend = c("Actual", "Fitted", "Predicted"), col = c("blue", "green", "red"), lty = 1)
