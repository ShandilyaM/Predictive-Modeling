# Import libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(rpart)
library(rpart.plot)
library(pROC)
library(e1071)
library(class)

# Load dataset
library(readr)
df1 <- read_csv("heart_disease.csv")
View(df1)
# Check structure of data
str(df1)
# Get summary of data
summary(df1)
# View first few rows of data
head(df1)
#Check for missing values
any(is.na(df1))
sapply(df1, function(x) sum(is.na(x)))

## Create new column Gender for Male and female
df2 <- as_tibble(df1)
df3<-df2 %>% 
  mutate(Gender=ifelse(Sex==0, "Female", "Male")) %>%
  mutate(HaveHeartDiseaseOrAttack=ifelse(HeartDiseaseorAttack==1,"Yes", "No"))

##Graph 1- Relation between people with high BP and heart disease by gender
# Assuming df1 contains columns HighBP, HeartDiseaseorAttack, and Sex

blood_pressure <- df3 %>%
  mutate(HaveHighBP = ifelse(HighBP == 0, "No", "Yes")) %>%
  group_by(HaveHighBP, Gender) %>%
  summarise(Total = sum(HeartDiseaseorAttack))

ggplot(blood_pressure, aes(x = HaveHighBP, y = Total, fill = Gender)) +
  geom_bar(stat = "identity", position = "dodge") + 
  labs(title = "Relationship between People with High Blood Pressure and Heart Disease by Gender",
       x = "People with High Blood Pressure",
       y = "Heart Disease or Attack case",
       fill = "Gender")


## Graph 2 - Relation between people with high chol and heart disease by gender
cholesterol <- df3 %>% 
  mutate(HaveHighChol = ifelse(HighChol == 0, "No", "Yes")) %>%
  group_by(HaveHighChol, Gender) %>%
  summarise(Total = sum(HeartDiseaseorAttack))

# Define colors for gender
gender_colors <- c("Male" = "blue", "Female" = "pink")

ggplot(cholesterol, aes(x = HaveHighChol, y = Total, fill = Gender)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Total), position = position_dodge(width = 0.9), vjust = -0.5, color = "black", size = 3) +  # Adding count labels
  scale_fill_manual(values = gender_colors) +  # Customizing colors
  labs(title = "Relationship between People with High Cholesterol and Heart Disease by Gender", 
       x = "People with High Cholesterol", 
       y = "Heart Disease or Attack case",
       fill = "Gender")

## Graph 3 - People with heart disease group by Age
## 1 2 3 4 5 6 7 8 9 10 11 12 13
factor(df1$Age)

age_range<-df1 %>%
  mutate(AgeRange=case_when(
    Age==1 ~ "18-24 yrs",
    Age==2 ~ "25-31 yrs",
    Age==3 ~ "32-38 yrs",
    Age==4 ~ "39-45 yrs",
    Age==5 ~ "46-52 yrs",
    Age==6 ~ "53-59 yrs",
    Age==7 ~ "60-66 yrs",
    Age==8 ~ "67-73 yrs",
    Age==9 ~ "74-80 yrs",
    Age==10 ~ "80-86 yrs",
    Age==11 ~ "87-93 yrs",
    Age==12 ~ "94-100 yrs",
    Age==13 ~ "101-107 yrs")) %>%
  group_by(AgeRange) %>%
  summarise(Total=sum(HeartDiseaseorAttack))
print(age_range)

ggplot(age_range, aes(x=AgeRange, y=Total)) +
  scale_x_discrete(limits=c("18-24 yrs",
                            "25-31 yrs",
                            "32-38 yrs",
                            "39-45 yrs",
                            "46-52 yrs",
                            "53-59 yrs",
                            "60-66 yrs",
                            "67-73 yrs",
                            "74-80 yrs",
                            "80-86 yrs",
                            "87-93 yrs",
                            "94-100 yrs",
                            "101-107 yrs")) +
  geom_bar(stat = "identity", fill=4) +
  geom_text(aes(label=Total),hjust=-0.5,colour="black") +
  coord_flip() +
  labs(title = "", 
       x="Age Group", 
       y="Heart Attack or Disease Case")

#correlation
view(cor(df1))
corrplot(cor(df1))

palette = colorRampPalette(c("blue", "white", "red")) (20)
heatmap(x = cor(df1), col = palette, symm = TRUE)

## Split dataset into training and testing sets
set.seed(123) # for reproducibility
index <- createDataPartition(df1$HeartDiseaseorAttack, p = 0.8, list = FALSE)
train_set <- df1[index,]
test_set <- df1[-index,]

## Build and predict models and get evaluation metrics

# Build Logistic Regression model
logit.reg <- glm(HeartDiseaseorAttack ~., data = train_set, family = "binomial")
options(scipen=999)
summary(logit.reg)

#Predict probabilities on test data 
logit.reg.pred <- predict(logit.reg, newdata = test_set, type = "response")  
# convert probabilities to binary predictions based on threshold 0.50
predicted <- ifelse(logit.reg.pred > 0.50, 1, 0)
# display summary and histogram of predictions
summary(logit.reg.pred)  
hist(logit.reg.pred) 
#Evaluating model
#confusion matrix 
con_mat <- table(Actual = test_set$HeartDiseaseorAttack, Predicted = predicted) 
print(con_mat)
#Check the accuracy by comparing actual vs predicted values and taking mean
accuracy <- mean(test_set$HeartDiseaseorAttack == predicted)
print(paste("Accuracy of Logistic Regression: ", round(accuracy * 100,2),"%"))

#ROC curve 

roc_logit <- roc(response = test_set$HeartDiseaseorAttack, predictor = as.numeric(logit.reg.pred)) 
plot.roc(roc_logit, print.thres = "best", col='blue', main = c("ROC Curve-Logistic Regression"), lty=1, lwd = 4) 

# Build Naive Bayes model

train_set$HeartDiseaseorAttack <- factor(train_set$HeartDiseaseorAttack)
test_set$HeartDiseaseorAttack <- factor(test_set$HeartDiseaseorAttack)

#check the distribution of target variable
table(train_set$HeartDiseaseorAttack)
table(test_set$HeartDiseaseorAttack)

levels_train <- levels(train_set$HeartDiseaseorAttack)
levels_test <- levels(test_set$HeartDiseaseorAttack)

if (!all(levels_test %in% levels_train)) {
  stop("Levels in test set do not match levels in training set")
}

model <- naiveBayes(HeartDiseaseorAttack ~ ., data = train_set)

#making predictions on test data
predictions <- predict(model, test_set)
# Convert both predictions and actual values to factors with the same levels 
predictions <- factor(predictions, levels = levels_train)
test_set$HeartDiseaseorAttack <- factor(test_set$HeartDiseaseorAttack, levels = levels_train)

#Evaluating model
confusionMatrix(predictions, test_set$HeartDiseaseorAttack)


# Build KNN model
# define predictor variables
predictors <- c("HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                "Diabetes", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
                "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
                "DiffWalk", "Sex", "Age", "Education", "Income")
k <- 5 # number of neighbors
knn_model <- knn(train = train_set[predictors], test = test_set[predictors],
                 cl = train_set$HeartDiseaseorAttack, k = k)

#Evaluating model
# Check the accuracy of the model
accuracy <- mean(knn_model == test_data$HeartDiseaseorAttack)
print(paste("Accuracy of KNN model:", round(accuracy * 100, 2), "%"))

# Compute predictions
predictions <- knn_model
# Create confusion matrix
confusion_matrix <- confusionMatrix(data = as.factor(predictions),
                                    reference = as.factor(test_set$HeartDiseaseorAttack))
print(confusion_matrix)
# Plot confusion matrix
plot(confusion_matrix$table,
     main = "Confusion Matrix for KNN Model",
     col = c("darkgreen", "darkred"),
     ylim = c(0, 1),
     cex = 1.5,
     xlab = "Predicted",
     ylab = "Actual")
