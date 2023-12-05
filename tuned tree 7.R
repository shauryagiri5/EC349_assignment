# Set-up, relevant libraries
install.packages('adabag')
install.packages('rpart') 
install.packages('tidytext')  
library(jsonlite)
library(adabag)
library(caret)
library(rpart)
library(rpart.plot)
library(tidytext)  

cat("\014")
rm(list=ls())

# Load data. Change for specific user
load("~/yelp_review_small.Rda")
load("C:/Users/shaur/Desktop/Year 3/Data Science/Project files/yelp_user_small.Rda")

merged_data <- merge(review_data_small, user_data_small, by = "user_id", all.x = TRUE)

features <- c("useful.x", "funny.x", "cool.x", "useful.y", "funny.y", "cool.y", "review_count", "text", "average_stars", "fans")
target <- "stars"  # Define your target variable
model_data <- merged_data[, c(features, target)]

bing_lexicon <- get_sentiments("bing")

positive_words <- bing_lexicon$word[bing_lexicon$sentiment == "positive"]
negative_words <- bing_lexicon$word[bing_lexicon$sentiment == "negative"]

analyze_sentiment <- function(text) {
  tokens <- tidytext::unnest_tokens(data.frame(text = text), output = word, input = text)
  
  sentiment_scores <- sum(tokens$word %in% positive_words) - sum(tokens$word %in% negative_words)
  
  positive_count <- sum(tokens$word %in% positive_words)
  negative_count <- sum(tokens$word %in% negative_words)
  
  return(c(sentiment_score = sentiment_scores, positive_count = positive_count, negative_count = negative_count))
}

count_words <- function(text) {
  words <- unlist(strsplit(as.character(text), " "))
  return(length(words))
}

model_data_no_missing <- model_data[complete.cases(model_data), ]

set.seed(1)
sampled_data <- model_data_no_missing[sample(nrow(model_data_no_missing), 100000), ]

sampled_data$text_count <- sapply(sampled_data$text, count_words)

sentiment_analysis_results <- t(sapply(sampled_data$text, analyze_sentiment))

sampled_data$sentiment_score <- sentiment_analysis_results[, "sentiment_score"]


sampled_data <- sampled_data[, !names(sampled_data) %in% c("text")]

set.seed(1)  
splitIndex <- caret::createDataPartition(sampled_data[, "stars"], p = 0.8, list = FALSE)
train_data <- sampled_data[splitIndex, ]
test_data <- sampled_data[-splitIndex, ]

train_data <- as.data.frame(train_data)
test_data <- as.data.frame(test_data)

tuning_grid <- expand.grid(
  cp = seq(0.001, 0.1, by = 0.001)
)

ctrl <- caret::trainControl(method = "cv", number = 10)  # 10-fold cross-validation

tuned_rpart_model <- caret::train(
  as.formula(paste(target, "~ .")),
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = tuning_grid
)

cat("Best CP value:", tuned_rpart_model$bestTune$cp, "\n")

pruned_rpart_model <- rpart::prune(tuned_rpart_model$finalModel, cp = tuned_rpart_model$bestTune$cp)

test_predictions_tuned_tree <- predict(pruned_rpart_model, newdata = test_data)

rmse_tuned_tree <- sqrt(mean((test_predictions_tuned_tree - test_data$stars)^2))
cat("Root Mean Squared Error (Tuned Tree):", rmse_tuned_tree, "\n")

rpart.plot(pruned_rpart_model, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)

r_squared <- caret::R2(pred = test_predictions_tuned_tree, obs = test_data$stars)
cat("R^2 Value:", r_squared, "\n")

summary(train_data)

cor_matrix <- cor(train_data)
print(cor_matrix)

linear_model <- lm(stars ~ ., data = train_data)
test_predictions_linear <- predict(linear_model, newdata = test_data)
rmse_linear <- sqrt(mean((test_predictions_linear - test_data$stars)^2))
cat("Root Mean Squared Error (Linear Regression):", rmse_linear, "\n")
r_squared_linear <- caret::R2(pred = test_predictions_linear, obs = test_data$stars)
cat("R^2 Value (Linear Regression):", r_squared_linear, "\n")

comparison_table <- data.frame(
  Model = c("Tuned Tree", "Linear Regression"),
  RMSE = c(rmse_tuned_tree, rmse_linear),
  R_squared = c(r_squared, r_squared_linear)
)
print(comparison_table)

stars_distribution <- table(sampled_data$stars)
barplot(stars_distribution, main = "Distribution of Stars", xlab = "Stars", ylab = "Frequency", col = "skyblue")
