# Install require packages.
install.packages("dplyr")
install.packages("ggplot2")
install.packages("e1071")
install.packages("nnet")
install.packages("pROC")

# Loading packages.
library(dplyr)
library(ggplot2)
library(e1071)
library(nnet)
library(pROC)
library(caret)

# Read the OpenPowerlifting dataset. 
powerlifting_data <- read.csv("./data/openpowerlifting-2022-07-07-4b6101fb.csv")

# Useful data filtered.
new_powerlifting_data <- select(powerlifting_data, Sex, Equipment, BodyweightKg, WeightClassKg, Best3SquatKg, Best3BenchKg, Best3DeadliftKg, TotalKg, Wilks, Place) %>% 
  mutate_all(list(~na_if(.,""))) %>% 
  filter(!(Place =="DQ" | Place == "DD" | Place == "NS" | Place == "G")) %>% 
  na.omit()
new_powerlifting_data$Place <- factor(ifelse(new_powerlifting_data$Place == 1, "Won", "Lost"))

# Split data between male and female because weight classes are different between the two.
female_powerlifting_data <- filter(new_powerlifting_data, Sex == "F")
male_powerlifting_data <- filter(new_powerlifting_data, Sex == "M") 

# Define weight classes.
female_weight_classes <- c(breaks = c(0, 47, 52, 57, 63, 69, 76, 84, Inf))
female_weight_class_labels <- c("47kg", "52kg", "57KG", "63kg", "69kg", "76kg", "84kg", "84kg+")
male_weight_classes <- c(breaks = c(0, 59, 66, 74, 83, 93, 105, 120, Inf))
male_weight_class_labels <- c("59kg", "66kg", "74KG", "83kg", "93kg", "105kg", "120kg", "120kg+")

# Replace original weight class column categorized to IPF standards.
female_powerlifting_data$WeightClassKg <- cut(female_powerlifting_data$BodyweightKg, 
                                              female_weight_classes, 
                                              labels = female_weight_class_labels)
male_powerlifting_data$WeightClassKg <- cut(male_powerlifting_data$BodyweightKg, 
                                            male_weight_classes, 
                                            labels = male_weight_class_labels)

# Put both datasets (male and female) into a list.
powerlifting_data_list <- list(male_powerlifting_data, female_powerlifting_data)
index_name <- c("Male", "Female") # Allows the category to be displayed on the plots through use of indexing.

# Visualize dataset
plot_sample <- new_powerlifting_data %>% sample_frac(0.005)
male_plot_sample <- male_powerlifting_data %>% sample_frac(0.005)
female_plot_sample <- female_powerlifting_data %>% sample_frac(0.005)

poly_plot <- ggplot(plot_sample, aes(x = BodyweightKg, y = TotalKg)) +
  geom_smooth(method = "lm", formula = y~poly(x, 3), se = FALSE, span = 0.2, colour = "black") +
  facet_wrap(~Sex)

female_box_plot  <- ggplot(female_plot_sample, aes(x = WeightClassKg, y = TotalKg)) +
  geom_boxplot() + 
  ggtitle("Female") + 
  xlab("Weight Class (kg)") + 
  ylab("Total Lifted (kg)")

male_box_plot  <- ggplot(male_plot_sample, aes(x = WeightClassKg, y = TotalKg)) +
  geom_boxplot() +
  ggtitle("Male") + 
  xlab("Weight Class (kg)") + 
  ylab("Total Lifted (kg)")

print(poly_plot)
print(female_box_plot)
print(male_box_plot)

# Program solution that displays chance of winning.
# Iterate through both categories; creating a model, validating, predicting, and then graphing the performance.
for (i in 1:length(powerlifting_data_list))
{
  # Create multinomial logistic regression model.
  sample <- sample.int(n = nrow(powerlifting_data_list[[i]]), size = floor(.75*nrow(powerlifting_data_list[[i]])), replace = F)
  logistic_train <- powerlifting_data_list[[i]][sample, ]
  logistic_test <- powerlifting_data_list[[i]][-sample, ]
  logistic_train$PlaceRef <- relevel(logistic_train$Place, ref = "Lost")
  logistic_model <- multinom(formula = PlaceRef ~ Equipment+WeightClassKg+Best3SquatKg+Best3BenchKg+Best3DeadliftKg+TotalKg+Wilks, data = logistic_train)
  
  # Validating the model.
  logistic_train$PredictedProbs <- predict(logistic_model, newdata = logistic_train, "probs")
  logistic_train$PredictedClass <- predict(logistic_model, newdata = logistic_train, "class")
  
  # Evaluate training data predictions to display validity .
  train_table <- table(logistic_train$Place, logistic_train$PredictedClass)
  print(paste0("Accuracy (Train): ", "(", index_name[[i]], ")", sprintf("%0.1f%%", round((sum(diag(train_table))/sum(train_table))*100, 2))))
  writeLines(paste0("Category (Train Confusion Matrix): ", index_name[[i]]))
  confusionMatrix(logistic_train$PredictedClass, logistic_train$Place)
  
  # Predicting for the test dataset.
  logistic_test$PredictedProbs <- predict(logistic_model, newdata = logistic_test, "probs")
  logistic_test$PredictedClass <- predict(logistic_model, newdata = logistic_test, "class")
  
  # Evaluate test data predictions.
  test_table <- table(logistic_test$Place, logistic_test$PredictedClass)
  print(paste0("Accuracy (Test): ", "(", index_name[[i]], ")", sprintf("%0.1f%%", round((sum(diag(test_table))/sum(test_table))*100, 2))))
  writeLines(paste0("Category (Test Confusion Matrix): ", index_name[[i]]))
  confusionMatrix(logistic_test$PredictedClass, logistic_test$Place)
  
  # Graph model's performance (ROC and AUC).
  multinom_roc <- roc(logistic_test$Place, logistic_test$PredictedProbs)
  multinom_auc <- round(auc(logistic_test$Place, logistic_test$PredictedProbs), 4)
  plot <- ggroc(multinom_roc, colour = "red", size = 2) +
  ggtitle(paste0("ROC Curve ", "(", index_name[[i]], ")", " (AUC = ", multinom_auc, ")"))
  print(plot)
}

# Program solution that can help a user increase their chances.
# User enters their 1 rep maxes for each lift, and their bodyweight.
bodyweight <- as.integer(readline(prompt="Enter bodyweight KG: "))
squat_1RM <- as.integer(readline(prompt="Enter 1RM squat KG: "))
benchpress_1RM <- as.integer(readline(prompt="Enter 1RM bench press KG: "))
deadlift_1RM <- as.integer(readline(prompt="Enter 1RM deadlift KG: "))

lifting_stats <-  data.frame(Best3SquatKg=c(squat_1RM), 
                             Best3BenchKg=c(benchpress_1RM),
                             Best3DeadliftKg=c(deadlift_1RM),
                             BodyweightKg=c(bodyweight))

# User can select output to predict a winning combined squat/bench press/deadlift, or bodyweight (that is based on the user's lifts).
writeLines("Select outcome:\n1. Squat/Bench Press/Deadlift\n2. Bodyweight")
lm_outcome_selection <- as.character(readline(prompt = "Input (1-2): "))

lm_outcome <- switch(lm_outcome_selection, "1" = c("Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"), "2" = c("BodyweightKg"))

lm_terms <- c("Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg", "BodyweightKg")

lm_variables <- unique(lm_terms[! lm_terms %in% lm_outcome])

lm_formula <- reformulate(lm_variables, sprintf("cbind(%s)", toString(lm_outcome)))

# Create multivariate linear regression model.
linear_model <- lm(lm_formula, data = powerlifting_data_list[[1]] %>% 
                     filter(Place == "Won") %>% 
                     filter(Equipment == "Raw"))

# Displays winning lifts or bodyweight based on user's lifting stats.
summary(linear_model)
predict(linear_model, newdata=lifting_stats)
