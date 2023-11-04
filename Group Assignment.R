# Load required libraries
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(ROSE)

# Suppress warnings
options(warn=-1)

# Read the CSV file into a dataframe
df <- read.csv("healthcare-dataset-stroke-data.csv")

# Display the first few rows of the dataframe
head(df)

# View information about the DataFrame
str(df)

# Generate summary statistics for the DataFrame
summary(df)

# Get the number of rows and columns in the DataFrame
dim(df)

# Replace missing values in 'bmi' column with 0
df$bmi[is.na(df$bmi)] <- 0

# Print the count of missing values for each column
print(colSums(is.na(df)))

# Display counts of unique values in the 'stroke' column
table(df$stroke)

# Filter the DataFrame for stroke patients
new_df <- df[df$stroke == 1, ]

# Calculate gender counts
gender_counts <- table(new_df$gender)

# Create a data frame for plotting
gender_data <- data.frame(gender = names(gender_counts), counts = gender_counts)

# Calculate percentages
gender_data$percentages <- (gender_data$counts / sum(gender_data$counts)) * 100

# Create a pie chart with percentages
ggplot(gender_data, aes(x = "", y = percentages, fill = gender)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(percentages, "%")), position = position_stack(vjust = 0.5)) +
  labs(title = "Gender Distribution for Stroke Patients") +
  theme_void() +
  theme(legend.position = "right")

# Filter the DataFrame for stroke patients
new_df2 <- df[df$stroke == 1, ]

# Calculate heart disease counts
heart_disease_counts <- table(new_df2$heart_disease)

# Create a data frame for plotting
heart_disease_data <- data.frame(heart_disease = names(heart_disease_counts), counts = heart_disease_counts)

# Calculate percentages
heart_disease_data$percentages <- (heart_disease_data$counts / sum(heart_disease_data$counts)) * 100

# Create a pie chart with percentages
ggplot(heart_disease_data, aes(x = "", y = percentages, fill = heart_disease)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(percentages, "%")), position = position_stack(vjust = 0.5)) +
  labs(title = "Heart Problem Distribution for Stroke Patients") +
  theme_void() +
  theme(legend.position = "right")

# Create a histogram with density curve for age
hist(new_df$age, breaks = seq(0, 100, by = 5), col = "grey",
     xlab = "Age", ylab = "Density", main = "Age of People Having Stroke", freq = FALSE)

# Add the density curve
lines(density(new_df$age), lwd = 2, col = 'red')

# Create a density plot for age with ggplot2
ggplot(new_df2, aes(x = age, fill = gender)) +
  geom_density(alpha = 0.5) +
  labs(x = "Age", y = "Density", title = "Age Distribution of People Having Stroke") +
  scale_fill_manual(values = c("Female" = "blue", "Male" = "red")) +
  theme_minimal()

# Create a histogram with density curve for average glucose level
hist(new_df$avg_glucose_level, col = "grey", xlab = "Average Glucose Level",
     ylab = "Density", main = "Average Glucose Level of People Having Stroke", freq = FALSE)

# Add the density curve
lines(density(new_df$avg_glucose_level), lwd = 2, col = 'red')

# Convert 'bmi' column to numeric
new_df$bmi <- as.numeric(new_df$bmi)

# Create a boxplot for the 'bmi' column
boxplot(new_df$bmi, main = "Variance of BMI", ylab = "BMI")

# Identify columns of type "character" or "factor"
char_factor_cols <- sapply(df, is.character) | sapply(df, is.factor)

# Create a list of column names that are of type "character" or "factor"
char_factor_col_names <- names(df[char_factor_cols])

# Select columns of data frame df with data type "character" or "factor"
d_list <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]

# Loop through each column in d_list and apply label encoding
for (i in d_list) {
  df[[i]] <- as.integer(factor(df[[i]]))
}

# Create feature set x1 with all columns except "stroke"
x1 <- new_df2[, !(names(new_df2) == 'stroke')]

# Select the 'stroke' column and assign it to y
y <- new_df2$stroke
# Set the seed for reproducibility
set.seed(0)

# Create a random sample of row indices for the testing set
test_indices <- sample(nrow(new_df2), 0.26 * nrow(new_df2))

# Split the data into training and testing sets
train_data <- new_df2[-test_indices, ]
test_data <- new_df2[test_indices, ]

# Separate the target variable (y) from the features (x)
x1_train <- train_data[, !names(train_data) %in% "stroke"]
y_train <- train_data$stroke
x1_test <- test_data[, !names(test_data) %in% "stroke"]
y_test <- test_data$stroke

# Combine the features and target variable into a single data frame
train_data <- data.frame(x1 = x1_train, y = y_train)

# Perform oversampling using the ROSE package
oversampled_data <- ROSE(y ~ ., data = train_data, seed = 0)$data

# Separate the oversampled data into features and target variable
x1_train_res <- oversampled_data[, !names(oversampled_data) %in% "y"]
y1_train_res <- oversampled_data$y

# Fit a logistic regression model
logistic_model <- glm(y_train_res ~ ., data = cbind(y_train_res, x1_train_res), family = binomial(link = "logit"))

# Predict on the test set
x1_test_data <- as.data.frame(x1_test)  # Ensure x_test is a data frame
pred <- predict(logistic_model, newdata =  x1_test_data, type = "response")

# Convert predicted probabilities to binary predictions (0 or 1)
pred <- ifelse(pred >= 0.5, 1, 0)

# Calculate the accuracy
accuracy <- sum(pred == y_test) / length(y_test)
print(accuracy)


