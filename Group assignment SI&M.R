# Loading required libraries
library(ggplot2)
library(dplyr)
library(caret)
library(ROSE)
library(randomForest)

# Loading and reading the dataset
df <- read.csv("healthcare-dataset-stroke-data.csv")
head(df)

# Replacing the  missing values in 'bmi' with 0
df$bmi[is.na(df$bmi)] <- 0

# Printing the number of missing values
print(colSums(is.na(df)))

# Exploring the 'stroke' variable
table(df$stroke)

# Exploratory data analysis
new_df <- subset(df, stroke == 1)

# Counting of genders in the dataset
gender_counts <- table(new_df$gender)
print(gender_counts)

# Creating a pie chart for gender, indicating "lightblue" for Female and "lightgreen" for Male
pie(gender_counts, labels = c("Female", "Male"), main = "Gender Distribution", 
    col = c("lightblue", "lightgreen"), border = "white")

# Counting of heart disease in the dataset
heart_disease_counts <- table(new_df$heart_disease)
print(heart_disease_counts)

# Creating a pie chart for heart disease, indicating 
pie(heart_disease_counts, labels = c("Yes", "No"), main = "Heart Problem Distribution", 
    col = c("orange", "brown"), border = "white")

# Creating histograms for age and avg_glucose_level for data distribution visualization
par(mfrow = c(1, 2))
hist(new_df$age, main = "Age Distribution for Stroke Patients", xlab = "Age", col = "pink")
hist(new_df$avg_glucose_level, main = "Avg Glucose Level for Stroke Patients", xlab = "Avg Glucose Level", col = "pink")


# To ensure BMI is treated as numeric
new_df$bmi <- as.numeric(new_df$bmi)

# Creating a boxplot for 'bmi' in order to provide valuable insights into the distribution
boxplot(new_df$bmi, main = "BMI Variance for Stroke Patients", col = "red")

# Encoding  categorical variables using LabelEncoder
categorical_cols <- sapply(df, is.character)
for (col in names(df)[categorical_cols]) {
  df[[col]] <- as.numeric(factor(df[[col]]))
}

# Splitting the data into training and testing sets in order to create model
set.seed(4)
train_indices <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
x_train <- df[train_indices, -which(names(df) == "stroke")]
y_train <- df[train_indices, "stroke"]
x_test <- df[-train_indices, -which(names(df) == "stroke")]
y_test <- df[-train_indices, "stroke"]

# Performing oversampling using SMOTE
oversampled_data <- ROSE(stroke ~ ., data = data.frame(x_train, stroke = y_train))$data

x_train_res <- oversampled_data[, -which(names(oversampled_data) == "stroke")]
y_train_res <- oversampled_data[, "stroke"]

# Logistic Regression Classifier
logistic_model <- glmnet(x = as.matrix(x_train_res), y = y_train_res, family = "binomial")

# Predicting the Logistic Regression model
predicted_probs_logistic <- predict(logistic_model, s = 0.01, newx = as.matrix(x_test), type = "response")
predicted_classes_logistic <- ifelse(predicted_probs_logistic > 0.5, 1, 0)

# Calculate accuracy for Logistic Regression
accuracy_logistic <- sum(predicted_classes_logistic == y_test) / length(y_test)
print(paste("Accuracy for Logistic Regression:", accuracy_logistic))

#  Random Forest models
num_models <- 5
models <- list()


for (i in 1:num_models) {
  
  subset_indices <- sample(1:nrow(x_train_res), size = 0.8 * nrow(x_train_res))
  x_train_subset <- x_train_res[subset_indices, ]
  y_train_subset <- y_train_res[subset_indices]
  
  # Training a Random Forest model on the dataset
  model <- randomForest(x_train_subset, y_train_subset, ntree = 100, mtry = 4)
  models[[i]] <- model
}

# Combining predictions from all models
predicted_classes_rf <- as.integer(rowMeans(sapply(models, function(model) {
  predict(model, newdata = x_test)
})))

# Calculating the  accuracy for Random Forest
accuracy_rf <- sum(predicted_classes_rf == y_test) / length(y_test)
print(paste("Accuracy for Random Forest:", accuracy_rf))


# Finding the Confusion Matrix for Random Forest
confusion_matrix_rf <- table(y_test, predicted_classes_rf)
print("Confusion Matrix:")
print(confusion_matrix_rf)

