# Load necessary libraries for various tasks
library(quantmod)    # For financial data analysis
library(readr)       # For data reading functions
library(pacman)      # For package management
library(xts)         # For time series analysis
library(zoo)         # For handling irregular time series
library(astsa)       # For time series analysis functions
library(nnet)        # For neural networks
library(stats)       # For statistical functions
library(RSNNS)       # For neural network functions
library(Rcpp)        # For C++ integration
library(TTR)         # For technical analysis functions
library(httr)        # For HTTP requests
library(binancer)    # For accessing Binance API
library(utils)       # For utility functions

# Get the Ethereum stock data from Binance
eth_usdt <- binance_klines('ETHUSDT', 
                           interval = '15m',
                           start_time = '2023-09-27',
                           end_time = '2023-10-2')

# Subset the data frame to include only the relevant columns
eth_usdt <- eth_usdt[, c("open_time", "open", "high", "low", "close")]

# Convert the subsetted data frame to an xts (time series) object
eth_usdt_ts <- as.ts(eth_usdt$open)

# Plot the Ethereum price time series
plot(eth_usdt_ts)

# Transform the Ethereum price data by taking the natural logarithm
y <- ts(log(eth_usdt_ts), frequency = 12)

# Perform time series decomposition (multiplicative) to separate components
decomp <- decompose(y, type = "multiplicative")

# Plot the decomposed time series components
plot(decomp)

# Plot the trend component of the decomposed time series
plot(decomp$trend, ylab = "Trend")

# Analyze the distribution of the residuals (random component)
random <- newdata <- na.omit(decomp$random)
plot(density(random))
qqnorm(random, pch = 1, frame = TRUE)
qqline(random, col = "steelblue", lwd = 2)

# Compute the autocorrelation function (ACF) and partial autocorrelation function (PACF)
acf(y)
pacf((y))

# Normalize the Ethereum price data to a range between 0 and 1
range_data <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
y <- range_data(eth_usdt_ts)
min_data <- min(eth_usdt_ts) 
max_data <- max(eth_usdt_ts)

# Convert the normalized data to a zoo object
y <- as.zoo(y)

# Create lagged variables (lags 1 to 12) of the normalized Ethereum price data
x1 <- Lag(y, k = 1)
x2 <- Lag(y, k = 2)
x3 <- Lag(y, k = 3)
x4 <- Lag(y, k = 4)
x5 <- Lag(y, k = 5)
x6 <- Lag(y, k = 6)
x7 <- Lag(y, k = 7)
x8 <- Lag(y, k = 8)
x9 <- Lag(y, k = 9)
x10 <- Lag(y, k = 10)
x11 <- Lag(y, k = 11)
x12 <- Lag(y, k = 12)

# Combine the lagged variables into a data frame
x <- cbind(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, y)
x <- cbind(x, y)

# Remove the first 12 rows to handle missing lagged values
x <- x[-(1:12),]

# Calculate the number of rows (n) in the final data frame
n = nrow(x)

# Set a random seed for reproducibility
set.seed(2018)

# Define the number of training samples
n_train <- 300

# Randomly sample training data indices
train <- sample(1:n, n_train, FALSE)

# Define inputs and outputs for the neural network
inputs <- x[, 2:13]
outputs <- x[, 1]

# Train an Elman neural network
num_input_features <- ncol(inputs)
fit <- elman(inputs, outputs, size = c(num_input_features, 2), maxit = 1000)

# Plot the iterative error during training
plotIterativeError(fit)

# Plot the regression error
plotRegressionError(outputs, fit$fitted.values)

# Calculate the squared correlation coefficient (R-squared) for training data
#a <- as.matrix(outputs[train])
#b <- as.matrix(fit$fitted.values)               
#cor_train <- cor(a, b)
#round(cor_train^2, 4)

# Make predictions on the test data
pred <- predict(fit, inputs[-train])

# Calculate the squared correlation coefficient (R-squared) for test data
#c <- as.matrix(outputs[train])
#d <- as.matrix(fit$fitted.values) 
#cor_test <- cor(c, d)
#round(cor_test^2, 4)

# Function to unscale data
unscale_data <- function(x, max_x, min_x) {
  (x * (max_x - min_x) + min_x)
}

# Unscale the actual output data for the test set
output_actual <- unscale_data(outputs[-train], max_data, min_data)

# Convert the unscaled actual output data into a matrix
output_actual <- as.matrix(output_actual)

# Set row names for the unscaled actual output data, useful for identification
rownames(output_actual) <- 1:length(output_actual)

# Unscale the predicted output data using the same scaling parameters
output_pred <- unscale_data(pred, max_data, min_data)

# Combine actual and predicted data
result <- cbind(as.ts(output_actual), as.ts(output_pred))

# Plot the actual vs. predicted data
plot(result[,1], type = "l", col = "blue", ylab = "Price")
lines(result[,2], col = "red")
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1)
