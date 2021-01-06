rm(list=ls())
setwd("C:/Users/abheer/Desktop/Data science/cp")

df <- read.csv("Womens_Clothing_E-Commerce_Reviews.csv", header=T)

names(df)
attach(df)
str(df)

################### Removing varaibles ###################

df1 <- df[,-c(1,2,4,5)]

df1$Recommended.IND <- as.factor(df1$Recommended.IND)

#################### Importing Libraries #####################

# Exploratory Data Analysis (EDA)

library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(GGally)
library(rpart)
library(ggplot2)

###################### Missing Value #####################

# MIssing value teartment 

# Treating missing values 
library(mice)
library(VIM)

# Displaying a graph to detect any missing data in the dataset
missmap(df1)

#################### Outlier treatment #####################

boxplot(df1$Age, horizontal = T)
bench1 = 52 + 1.5 * IQR(df1$Age)
bench1
df1$Age[df1$Age > bench1] <- bench1
boxplot(df1$Age, horizontal = T)

boxplot(df1$Positive.Feedback.Count, horizontal = T)
bench2 = 3 + 1.5 * IQR(df1$Positive.Feedback.Count)
bench2
df1$Positive.Feedback.Count[df1$Positive.Feedback.Count > bench2] <- bench2
boxplot(df1$Positive.Feedback.Count, horizontal = T)

##################### Multicollienrity ######################

# Creating a seperate dataset to check multicollinearity 
library(faraway)

df_MC <- df1

# Changing variable types for the test
df_MC$Recommended.IND <- as.integer(df_MC$Recommended.IND)
df_MC$Division.Name <- as.integer(df_MC$Division.Name)
df_MC$Department.Name <- as.integer(df_MC$Department.Name)
df_MC$Class.Name <- as.integer(df_MC$Class.Name)

mymodel = lm(Recommended.IND ~ ., data = df_MC)

vif(mymodel)

# No evidence of multicollinearity was found 
# The VIF value for all the variables was less than 4 

#################### Renaming variables #################

# Change the name of the following varaibles
library(reshape)

df1 <- rename(df1, c( Recommended.IND = 'Recommended'))
df1 <- rename(df1, c( Division.Name = 'Division'))
df1 <- rename(df1, c( Department.Name = 'Department'))
df1 <- rename(df1, c( Class.Name = 'Class'))
df1 <- rename(df1, c( Positive.Feedback.Count = 'Feedback_Count'))
df1 <- rename(df1, c( Review.Text = 'Review_Text'))

str(df1)
attach(df1)
################ Univariate Analysis ####################

# Develop histogram of Age
hist(Age, col = "Red")

# Develop histogram of Rating
hist(Rating, col = "Blue")

# Develop histogram of Rating
hist(FeedbacKCount, col = "Green")

################# Bivariate Analysis ####################

# Understanding the correlation between independent and 
# dependent varaible (Recommended)

plot(Recommended, Age)
plot(Recommended, Rating)
plot(Recommended, Feedback_Count)

##########################################################

ggplot(data = df1, mapping = aes(x = Age)) +
  geom_freqpoly(mapping = aes(color = Class), binwidth = 500)

##################### Train Test Split ###################

set.seed(42)
ind <- createDataPartition(df1$Recommended, p = 8/10, list = FALSE)
traindf <- df1[ind,]
testdf <- df1[-ind,]

str(traindf)
summary(traindf)

##################################################################
############ K-Fold Tuning for Logistic Regression ###############
##################################################################

logit_control <- trainControl(method = "cv",
                           number = 5,
                           search = "random",
                           savePredictions = T)

  
logit_fitcv <- train(Recommended~.,
                   data = na.exclude(traindf),
                   method = "glm",
                   family = "binomial",
                   trControl = logit_control)


summary(logit_fitcv)

logit_fitcv

caret::confusionMatrix(table((logit_fitcv$pred)$pred, 
                             (logit_fitcv$pred)$obs))

# Developing a new model with only the varaibles > 95% 
# significance levels 

logit_fitcv_sig <- train(Recommended ~ Age + Rating +
                           Feedback_Count,
                   data = na.exclude(traindf),
                   method = "glm",
                   family = "binomial",
                   trControl = logit_control)

summary(logit_fitcv_sig)

logit_fitcv_sig

caret::confusionMatrix(table((logit_fitcv_sig$pred)$pred, 
                             (logit_fitcv_sig$pred)$obs))

Logistic_model <- glm(Recommended ~ Age + Rating +
                        Feedback_Count,
                        data = traindf, 
                        family=binomial)

summary(Logistic_model)

plot(as.factor(Logistic_model$y), Logistic_model$fitted.values)

# As shown in the model the boxplot has a very high distinctive and 
# predictive power as the boxplots differ in a larger manner 

res <- predict(Logistic_model, testdf,
               type = "response")

##################### Confusion Matrix ########################

table(ActualValue = testdf$Recommended, 
                      PredictedValue = res > 0.5)

#################### Optimizing Threshold ######################

library(ROCR)

ROCR_Pred <- prediction(res, testdf$Recommended)
ROCR_Pref <- performance(ROCR_Pred, "tpr", "fpr")

plot(ROCR_Pref, colorize = T, print.cutoffs.at = seq(0.1, by = 0.1))

############ Re-configuring the Threshold #######################

table(ActualValue = testdf$Recommended, 
      PredictedValue = res > 0.6)

##### Accuarcy ######
(791+3611)/(791+3611+249+43)

# 93.77% from 93.67%

##### Sensitivity ######
3611/(3611+249)

# 93.54%

##### Specificity ######
791/(791+43)

# 94.84%

##################################################################
############### K-Fold Tuning for Random Forest ##################
##################################################################


# Hyperparametr tuning Random Forest

fitcontrol <- trainControl(method = "cv",
                           number = 5,
                           search = "random",
                           savePredictions = T)

fitcontrol_repeated <- trainControl(method = "repeatedcv",
                                    number = 5,
                                    search = "random",
                                    repeats = 3,
                                    savePredictions = T)

# We can use fitcontrol or fitcontrol_repeated for repeated cv

rf_fit_cv <- train(Recommended~.,
                   data = na.exclude(traindf),
                   method = "rf",
                   trControl = fitcontrol,
                   tuneLength = 10,
                   ntree = 100)

rf_fit_cv$bestTune

# Ploting variance importance plot 
# It gives the most import varaible 
plot(varImp(rf_fit_cv, scale = F), main = "Var Imp RF 5 fold cv")

####### Developing a confusion matrix with best parameters ######

Optimal_rf_ = subset(rf_fit_cv$pred, rf_fit_cv$pred$mtry == 
                  rf_fit_cv$bestTune$mtry)

caret::confusionMatrix(table(Optimal_rf_$pred,Optimal_rf_$obs))

#################################################################

###### Initial EDA ########

nrow(traindf)
sum(traindf$Recommended == "1")/nrow(traindf)

######  #######

library(randomForest)
set.seed(100)

rnd_Forest <- randomForest(Recommended ~., 
                           data = traindf,
                           ntree = 501,
                           mtry = 3,
                           nodesize = 10,
                           importance = TRUE)

print(rnd_Forest)
# What does OBB determine ?

# Printing the error rate drecrease along vs no. of trees
print(rnd_Forest$err.rate)

# Ploting the error rate drecrease along vs no. of trees
plot(rnd_Forest)

# Determining the important parameters
importance(rnd_Forest)

# Tuning random forest

set.seed(100)
tRnd_Forest <- tuneRF(x = traindf[,-c(3)], 
                      y = traindf$Recommended,
                      mtryStart = 3,
                      stepFactor = 1.5,
                      ntreeTry = 51,
                      improve = 0.0001,
                      nodesize = 10,
                      trace = TRUE,
                      plot = TRUE,
                      doBest = TRUE,
                      importance = TRUE)

# Incorporating a predicted class and their probabilities column
traindf$predict_class <- predict(tRnd_Forest, traindf, type = "class")
traindf$prob_of_1 <- predict(tRnd_Forest, traindf, type = "prob")[,"1"]
head(traindf)

nrow(traindf)

# Developing a table to determine error rate
tbl <- table(traindf$Recommended, traindf$predict_class)
print(( tbl[1,2] + tbl[2,1] ) / 18778 )

# Dividing the data into 10 quantiles based on probabilities 
# based of them recommmending the services to others 
qs <- quantile(traindf$prob_of_1, probs = seq(0, 1, length = 11))
print(qs)

# Determining accuracy of the data
threshold <- 0.99
mean(traindf$Recommended[traindf$prob_of_1 > threshold] == "1")

# Fitting the tuned random forest to test dataset to get probabilities
testdf$predict_class <- predict(tRnd_Forest, testdf, type = "class")
testdf$prob_of_1 <- predict(tRnd_Forest, testdf, type = "prob")[,"1"]
head(testdf)

nrow(testdf)

# Developing a table to determine error rate in the test dataset
tbl <- table(testdf$Recommended, testdf$predict_class)
print(( tbl[1,2] + tbl[2,1] ) / 4694 )

# Determining accuarcy for test data
mean(testdf$Recommended[testdf$prob_of_1 > threshold] == "1")

# Optimize the nodesize to eliminate overfitting 

##################################################################
####################### Decision tree / CART  ####################
##################################################################

library(e1071)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

Cart_control <- trainControl(method = "cv",
                           number = 5,
                           search = "random",
                           savePredictions = T)

cart_grid <- expand.grid(.cp = (0:20)*0.001)

Cart_fit_cv <- train(Recommended~.,
                   data = na.exclude(traindf),
                   method = "rpart",
                   trControl = Cart_control,
                   tuneGrid = cart_grid)

Cart_fit_cv

plot(Cart_fit_cv)

# Developing a tree using complexity parameter with lowest error 

tree_rp <- rpart(Recommended ~ ., 
                 data = na.exclude(traindf),
                 method = "class",
                 control = rpart.control(cp = 0.001))

tree_rp <- rpart(Recommended ~ ., 
                 data = na.exclude(traindf),
                 method = "class",
                 cp = 0.001)

fancyRpartPlot(tree_rp, caption = NULL)

rpart.plot(tree_rp)

# Predicting the data
tree_predictions <- predict(tree_rp, testdf, type = "class")

# Confusion matrix
confusionMatrix(tree_predictions, testdf$Recommended)

# Visualizing a decision tree
prp(tree_rp)

##############################################################
############### Sentimental Analysis #########################
##############################################################

library(tidyr)

# Concatinating the Title and Reveiws columns into single column 
df1 <- unite(df, "Title&Reviews", Title, Review.Text, 
             sep = " ", remove = T)
names(df1)

# Building a corpus

# Importing text mining library
library(tm)

Reviews_corpus <- iconv(df1$`Title&Reviews`, to = "UTF-8")
R_corpus <- Corpus(VectorSource(Reviews_corpus))
inspect(R_corpus[1:5])

# Clean text
r_corpus <- tm_map(R_corpus, tolower)
inspect(r_corpus[1:5])
r_corpus <- tm_map(r_corpus, removePunctuation)
inspect(r_corpus[1:5])
r_corpus <- tm_map(r_corpus, removeNumbers)
inspect(r_corpus[1:5])
r_corpus <- tm_map(r_corpus, removeWords, stopwords('english'))
inspect(r_corpus[1:5])

# r_corpus <- tm_map(r_corpus, stemDocument, language = "english")
# inspect(r_corpus[1:5])

# Removing url 
remove_url <- function(x) gsub('http[[:alnum:]]*', '', x)
cleanset <- tm_map(r_corpus, content_transformer(remove_url))
inspect(cleanset[1:5])

# Removing username
remove_username <- function(x) gsub('@', '', x)
cleanset <- tm_map(r_corpus, content_transformer(remove_username))
inspect(cleanset[1:5])

cleanset <- tm_map(cleanset, stripWhitespace)
inspect(cleanset[1:5])



# Term Document Matrix
tdm <- TermDocumentMatrix(cleanset)
tdm
tdm <- as.matrix(tdm)
tdm[1:10,1:10]

# Checking the dimension of the tdm matrix
dim(tdm)


# Creating the Term Document Matrix to remove sparse terms 
tdm <- TermDocumentMatrix(cleanset)


##################################################################

# Remove sparse terms that occur in less 96% of the documents 
# This is an effective way to remove outliers 
sparse_96 <- removeSparseTerms(tdm, 0.96)
sparse_96
dim(sparse_96)

# After removing sparse terms we get 183 terms that
sparse1_96 <- as.matrix(sparse_96)
sparse1_96[1:10,1:10]

# Barplot 
w_96 <- rowSums(sparse1_96)
w_96 <- subset(w_96 , w_96 >= 500)
barplot(w_96, las = 2, col = rainbow(50))
word_freq_96 <- data.frame(term = names(w_96), freq = w_96)
word_freq_96

##################################################################

# Remove sparse terms that occur in less 95% of the documents 
# This is an effective way to remove outliers 
sparse_95 <- removeSparseTerms(tdm, 0.95)
sparse_95
dim(sparse_95)

# After removing sparse terms we get 183 terms that
sparse1_95 <- as.matrix(sparse_95)
sparse1_95[1:10,1:10]

# Barplot 
w_95 <- rowSums(sparse1_95)
w_95 <- subset(w_95, w_95 >= 500)
barplot(w_95, las = 2, col = rainbow(50))
word_freq_95 <- data.frame(term = names(w_95), freq = w_95)
word_freq_95

##################################################################

# Remove sparse terms that occur in less 92% of the documents 
# This is an effective way to remove outliers 
sparse_92 <- removeSparseTerms(tdm, 0.92)
sparse_92
dim(sparse_92)

# After removing sparse terms we get 183 terms that
sparse1_92 <- as.matrix(sparse_92)
sparse1_92[1:10,1:10]

# Barplot 
w_92 <- rowSums(sparse1_92)
w_92 <- subset(w_92, w_92 >= 500)
barplot(w_92, las = 2, col = rainbow(50))
word_freq_92 <- data.frame(term = names(w_92), freq = w_92)
word_freq_92

##################################################################

# Remove sparse terms that occur in less 97% of the documents 
# This is an effective way to remove outliers 
sparse_97 <- removeSparseTerms(tdm, 0.97)
sparse_97
dim(sparse_97)

# After removing sparse terms we get 183 terms that
sparse1_97 <- as.matrix(sparse_97)
sparse1_97[1:10,1:10]

# Barplot 
w_97 <- rowSums(sparse1_97)
w_97 <- subset(w_97, w_97 >= 500)
barplot(w_97, las = 2, col = rainbow(50))
word_freq_97 <- data.frame(term = names(w_97), freq = w_97)
word_freq_97

##################################################################

library(wordcloud)
x <- sort(rowSums(sparse1_97), decreasing = T)
set.seed(123)
wordcloud(words = names(x),
          freq = x,
          max.words = 150,
          random.order = F,
          colors = brewer.pal(8, 'Dark2'),
          scale = c(3, 0.3),
          rot.per = 0.2)

##################################################################

library(tm)
list1 <- findAssocs(tdm, "short", 0.1)
corr_df1 <- t(data.frame(t(sapply(list1, c))))
corr_df1

barplot(t(as.matrix(corr_df1)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "blue", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list2 <- findAssocs(tdm, "retailer", 0.09)
corr_df2 <- t(data.frame(t(sapply(list2, c))))
corr_df2

barplot(t(as.matrix(corr_df2)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "red", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list3 <- findAssocs(tdm, "dress", 0.1)
corr_df3 <- t(data.frame(t(sapply(list3, c))))
corr_df3

barplot(t(as.matrix(corr_df3)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list4 <- findAssocs(tdm, "love", 0.075)
corr_df4 <- t(data.frame(t(sapply(list4, c))))
corr_df4

barplot(t(as.matrix(corr_df4)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list5 <- findAssocs(tdm, "sweater", 0.075)
corr_df5 <- t(data.frame(t(sapply(list5, c))))
corr_df5

barplot(t(as.matrix(corr_df5)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")


##################################################################

list6 <- findAssocs(tdm, "material", 0.075)
corr_df6 <- t(data.frame(t(sapply(list6, c))))
corr_df6

barplot(t(as.matrix(corr_df6)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list7 <- findAssocs(tdm, "shirt", 0.06)
corr_df7 <- t(data.frame(t(sapply(list7, c))))
corr_df7

barplot(t(as.matrix(corr_df7)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list8 <- findAssocs(tdm, "fabric", 0.08)
corr_df8 <- t(data.frame(t(sapply(list8, c))))
corr_df8

barplot(t(as.matrix(corr_df8)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list9 <- findAssocs(tdm, "price", 0.08)
corr_df9 <- t(data.frame(t(sapply(list9, c))))
corr_df9

barplot(t(as.matrix(corr_df9)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list10 <- findAssocs(tdm, "sale", 0.08)
corr_df10 <- t(data.frame(t(sapply(list10, c))))
corr_df10

barplot(t(as.matrix(corr_df10)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")

##################################################################

list11 <- findAssocs(tdm, "fit", 0.08)
corr_df11 <- t(data.frame(t(sapply(list11, c))))
corr_df11

barplot(t(as.matrix(corr_df11)), beside = TRUE, xlab = "words",
        ylab = "correlation", col = "yellow", 
        main = "Fabric Correlation with other words",
        border = "black")




# Sentiment Analysis 
library(syuzhet)
library(lubridate)
library(scales)
library(reshape2)
library(dplyr)
library(ggplot2)

# Reading file
Reviews_corpus <- iconv(df1$`Title&Reviews`, to = "UTF-8")

# Obtaining sentiment scores
s <- get_nrc_sentiment(Reviews_corpus)
head(s)
Reviews_corpus[6]

# Barplot
barplot(colSums(s),
        las = 2,
        col = rainbow(10),
        ylab = 'Count',
        main = 'Sentiment Scores for Women Clothing Reviews')





#################################################################
# 5 Star Rating

library(tidytext)
library(itunesr)
library(tidyverse)

ecom_reviews_5 <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Rating == 5],
                           stringsAsFactors = FALSE)

ecom_reviews_5 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_5 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "Green") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for 5 star Rating",
       caption = " Bigram Freq")

#################################################################

ecom_reviews_4 <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Rating == 4],
  stringsAsFactors = FALSE)

ecom_reviews_4 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_4 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for 4 star Rating",
       caption = "Bigram freq")


#################################################################

ecom_reviews_3 <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Rating == 3],
  stringsAsFactors = FALSE)

ecom_reviews_3 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_3 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for 3 star Rating",
       caption = "Bigram freq")

#################################################################

ecom_reviews_2 <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Rating == 2],
  stringsAsFactors = FALSE)

ecom_reviews_2 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_2 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for 2 star Rating",
       caption = "Bigram freq")

#################################################################

ecom_reviews_1 <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Rating == 1],
  stringsAsFactors = FALSE)

ecom_reviews_1 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_1 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for 1 star Rating",
       caption = "Bigram freq")

#################################################################

ecom_reviews_Not_Recommended <- data.frame(
  txt = s_df1$`Title&Reviews`[s_df1$Recommended.IND == 0],
  stringsAsFactors = FALSE)

ecom_reviews_Not_Recommended %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_Not_Recommended %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for where customers will not recommend",
       caption = "Bigram freq")

#################################################################
####################### Genearl Division ########################
#################################################################

review_General <- filter(s_df1, s_df1$Division.Name == "General")
review_General <- filter(s_df1, s_df1$Rating == 1)

ecom_reviews_1 <- data.frame(
  txt = review_General$`Title&Reviews`[review_General$Recommended.IND == 0],
  stringsAsFactors = FALSE)

ecom_reviews_1 %>% 
  unnest_tokens(output = word, input = txt) %>% 
  anti_join(stop_words) %>% 
  count(word, sort = TRUE) 

ecom_reviews_1 %>% 
  unnest_tokens(word, txt, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE) %>% 
  slice(1:10) %>% 
  ggplot() + geom_bar(aes(word, n), stat = "identity", fill = "#de5833") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top Bigrams for where customers will not recommend",
       caption = "Division: General and Rating: 1")

