# Input Variable (x) - Other Factors
# Output Variable(y) - Salary

# Load the Dataset

# Training Data - Data file is imported by Text(base) to convert strings into factors
Salary_train <- read.csv(file.choose())
View(Salary_train)
str(Salary_train)
attach(Salary_train)

Salary_train$educationno <- as.factor(Salary_train$educationno)
class(Salary_train)

# Test Data - Data file is imported by Text(base) to convert strings into factors
Salary_test <- read.csv(file.choose())
View(Salary_test)
str(Salary_test)


Salary_test$educationno <- as.factor(Salary_test$educationno)
class(Salary_test)


# Exploratory Data Analysis

summary(Salary_train)
summary(Salary_test)

# Graphical Visualization 

# Plot

plot(workclass,Salary, main = "Workclass")
plot(education,Salary, main = "Education")
plot(occupation,Salary, main = "Occupation")
plot(relationship,Salary, main = "Relationship")


# ggplot

library(ggplot2)

ggplot(data= Salary_train,aes(x=Salary, y = age, fill = Salary)) +
  geom_boxplot() + ggtitle("Box Plot")

ggplot(data=Salary_train,aes(x=Salary, y = hoursperweek, fill = Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")

#Density Plot 

ggplot(data=Salary_train,aes(x = age, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = workclass, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = education, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = educationno, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')



# proportion of salary
prop.table(table(Salary_test$Salary))
prop.table(table(Salary_train$Salary))

##  Training a model on the data ----
install.packages("e1071")
library(e1071)

# Naive Bayes Model 
Model <- naiveBayes(Salary_train$Salary ~ ., data = Salary_train)
Model

##  Evaluating model performance
Salary_test_pred <- predict(Model, Salary_test)

library(gmodels)
CrossTable(Salary_test_pred, Salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

test_acc = mean(Salary_test_pred == Salary_test$Salary)
test_acc

# On Training Data
Salary_train_pred <- predict(Model, Salary_train)

train_acc = mean(Salary_train_pred == Salary_train$Salary)
train_acc

