################################################################################
# MovieLens Project
# Scott Rushford
###############################################################################

# EXECUTIVE SUMMARY
# The object of this project is to develop a recommender model that produces a
# RMSE less than 0.86490
# |method                                |      RMSE|
##|:-------------------------------------|---------:|
##|Just the average                      | 1.0482202|
##|Movie Effect Model                    | 0.9862839|
##|Movie + User Effects Model            | 0.9077043|
##|Regularized Movie Effect Model        | 0.9649457|
##|Regularized Movie + User Effect Model | 0.8806419|

# Project Data
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Exploratory Data Analysis
## Review data
head(edx)
str(edx)

## Summarize Data
### Distinct users and movies
edx_sum <- edx %>% 
     mutate(n_users = n_distinct(userId), n_movies = n_distinct(movieId),
            avg = mean(rating))

### Average Rating
mean(edx$rating)

### Review Users and movies
reviews_user <- edx %>% 
  select(userId, movieId) %>% 
  group_by(userId) %>% 
  summarise(total_reviews = n ())
mean(reviews_user$total_reviews)
max(reviews_user$total_reviews)
which.max(reviews_user$total_reviews)

### Review Movies
reviews_movie <- edx %>% 
  select(userId, movieId, title) %>% 
  group_by(movieId, title) %>% 
  summarise(total_reviews = n ())
mean(reviews_movie$total_reviews)
max(reviews_movie$total_reviews)
which.max(reviews_movie$total_reviews)

# Summary 69878 users rating 10677 movies with a total number of ratings of
# 9000055 avg rating 3.51 each user rating on average 128.8 movies. 
# On average each movie is rated 843 times with the max being Pulp Fiction
# at 31362 reviews.

# Load the other required packages 
if(!require(recommenderlab)) 
  install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(knitr)) 
  install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) 
  install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) 
  install.packages("tinytex", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) 
  install.packages("corrplot", repos = "http://cran.us.r-project.org")

library(recommenderlab)
library(Matrix)
library(knitr)
library(tinytex)
library(corrplot)

#PCA is timestamp a useful variable

edx_pca <- edx %>% 
  select(userId, movieId, rating, timestamp)

# Normalize the data
pca_norm <- scale(edx_pca)
  
# PCA
pca <- prcomp(pca_norm)
summary(pca)
plot(pca)

ca <- cor(pca_norm)
corrplot(ca)

# Timestamp has a correlation with moving rating at 0.374. This is explained by
# the fact that the movie ID is assigned as the movie is released. It is
# expected that a movie will receive more ratings after it is released.


# use the collaborative filtering algorithms available in recommenderlab
# there are two methods in recommenderlab to create models
# the ratings method is used to create models and not the topnlist method so
# title is represented by movieID
# the title column can be eliminated and it will speed up the processing of the 
# algorithms
# In order to use the algorithms from recommeberlab the data needs to be in
# a realRatingMatrix
# Remove time stamp, title and genre.
## Coerce data into a realRatingMatrix

edx_1 <- edx %>% 
  select(userId, movieId, rating)

which(is.na(edx_1), arr.ind = TRUE)

edx_r <- as(edx_1, "realRatingMatrix")

rm(edx_1)


str(edx)
# Data Visualization 

# Histogram of Ratings
edx %>% ggplot(aes(rating)) +
  geom_histogram(bins = 10, col = "black", fill = "red") +
  scale_x_log10() +
  geom_vline(xintercept = mean(edx$rating), linewidth = 2) +
  xlab("Rating Mean 3.51") +
  ggtitle("Rating Histogram and Mean")




# Visualize the number of time a movie was rated and the number of 
# times a user rated a movie.
if(!require(gridExtra)) 
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
library(gridExtra)

p1 <- edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "red") + 
  scale_x_log10() +
  xlab("Movies by Number of Times Rated")
  ggtitle("Movies")

p2 <- edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "red") + 
  scale_x_log10() + 
  xlab("Users by Number of Movies Rated") +
  ggtitle("Users")

grid.arrange(p1, p2, ncol = 2)

# Examine Sparsity  - how frequently is a movie rated first 500 users
spars1 <- image(edx_r [1:500, 1:500])
spars1

# To reduce dimensions remove movies that have not been rated at least 843
# times and users that have not rated at least 129 movies. Based on averages
# produced in the summary analysis
edx_r <- edx_r[rowCounts(edx_r) >= 129,
               colCounts(edx_r) >= 843]
edx_r
spars2 <- image(edx_r [1:500, 1:500])
# The matrix is reduced to a 24115 x 2195 rating matrix with 5796726 ratings.


## Use similarity matrices to see if
## there is greater similarity between users or items. From video (@ the outlier 73)
user_similarity <- similarity(edx_r[1:10, ], method = "cosine", 
                              which = "users")
user_sim <- as.matrix(user_similarity)

movie_similarity <- similarity(edx_r[ , 1:10], method = "cosine", 
                              which = "items")
movie_sim <- as.matrix(movie_similarity)

image(user_sim [1:10, 1:10])
image(movie_sim [1:10, 1:10])


# Check the available algorithms in recommenderlab
recommenderRegistry$get_entries(dataType = "realRatingMatrix")


# Create an evaluation scheme to split the data into training, test and
# validation sets using Cross Validation k = 10. As the average rating 
# is 3.51 a good rating is considered anything greater or equal to 4.
# for given use the four experimental with holding protocols called
# Given 2, Given 5, Given 10 and All-but-1. @Hahsler

set.seed(76)
es_edx_r <-  evaluationScheme(edx_r, method = "cross-validation", given = 25,
                              goodRating = 4)
es_edx_r


# "evaluationScheme() creates 3 data sets. It splits the data into train and 
#  test set but then within the test set it further creates a known and an
#  unknown data sets. The known test data has the ratings specified by given 
#  and unknown has the remaining ratings, which will be used to validate the 
#  predictions made using known." @Malshe

t_edx_r <-  getData(es_edx_r, "train")
k_edx_r <-  getData(es_edx_r, "known")
u_edx_r <-  getData(es_edx_r, "unknown")

str(t_edx_r)
str(k_edx_r)
str(u_edx_r)
t_edx_r
k_edx_r
u_edx_r

# Remove unnecessary items from the environment to free up memory.
rm(edx)
rm(edx_r)
rm(es_edx_r)

# t_edx_r
#21699 x 2195 rating matrix of class ‘realRatingMatrix’ with 5218067 ratings.
#> k_edx_r
#2416 x 2195 rating matrix of class ‘realRatingMatrix’ with 48003 ratings.
#> u_edx_r
#2416 x 2195 rating matrix of class ‘realRatingMatrix’ with 530656 ratings.



# WARNING - depending on your processor speed these alogrithms will take
# several minutes and possibly up to an hour to complete.



# Popular establish baseline - what if we only recommended popular movies
# 
# Train Popular
POP_rec <- Recommender(t_edx_r, method = "POPULAR")

# Evaluate Popular
POP_eval <- POP_rec %>% 
  predict(k_edx_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_edx_r)

print(POP_eval)

# Remove unnecessary items from the environment to free up memory.

rm(POP_rec)

# IBCF
# Train IBCF model
set.seed(4)
IBCF_rec <- Recommender(t_edx_r, method = "IBCF")

# IBCF evaluate model
IBCF_eval <- IBCF_rec %>% 
  predict(k_edx_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_edx_r)

print(IBCF_eval)

# Remove unnecessary items from the environment to free up memory.

rm(IBCF_rec)

# UBCF
# Train UBCF model
set.seed(6)
UBCF_rec <- Recommender(t_edx_r, method = "UBCF")

# UBCF model evaluate
UBCF_eval <- UBCF_rec %>% 
  predict(k_edx_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_edx_r)

print(UBCF_eval)

# Remove unnecessary items from the environment to free up memory.
rm(UBCF_rec)

# SVD
# SVD model train
set.seed(1056)
SVD_rec <- Recommender(t_edx_r, method = "SVD")

# SVD model evaluate
SVD_eval <- SVD_rec %>% 
  predict(k_edx_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_edx_r)

print(SVD_eval)

# Remove unnecessary items from the environment to free up memory.
rm(SVD_rec)

# SVDF
# SVDF model train
set.seed(1100)
SVDF_rec <- Recommender(t_edx_r, method = "SVDF")

# SVDF Evaluation
SVDF_eval <- SVDF_rec %>% 
  predict(k_edx_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_edx_r)

print(SVDF_eval)


# Remove unnecessary items from the environment to free up memory.
rm(SVDF_rec)

# Evaluate Models
model_eval <- data.frame(Model=c('IBCF', 'UBCF', "POP", "SVD", "SVDF"),
                         RMSE=c(IBCF_eval[1], UBCF_eval[1], POP_eval[1], 
                                SVD_eval[1], SVDF_eval[1]))



em <-  model_eval %>% 
  ggplot(aes(reorder(Model,RMSE), RMSE)) +
  geom_bar(stat = 'identity', col = 'black', fill = 'red') +
  labs(x = 'Model' , y = 'RMSE', title = 'Model Accuracy')
em

# SVDF has the lowest RMSE but takes longer to run, Popular and SVD are quicker
# to run with only a modest reduction in accuracy
# Remover unnecessary files
rm(k_edx_r, t_edx_r, u_edx_r)

# Testing on Final Holdout
# Compare final holdout to edx mean rating, unique users/movies etc.
## Summarize Data
### Distinct users and movies (course)
final_holdout_test %>% 
     summarize(n_users = n_distinct(userId),
               n_movies = n_distinct(movieId))

### Average Rating
mean(final_holdout_test$rating)

### Review Users
reviews_userfht <- final_holdout_test %>% 
  select(userId, movieId) %>% 
  group_by(userId) %>% 
  summarise(total_reviews = n ())
mean(reviews_userfht$total_reviews)
max(reviews_userfht$total_reviews)
which.max(reviews_userfht$total_reviews)

### Review Movies
reviews_moviefht <- final_holdout_test %>% 
  select(userId, movieId, title) %>% 
  group_by(movieId, title) %>% 
  summarise(total_reviews = n ())
mean(reviews_moviefht$total_reviews)
max(reviews_moviefht$total_reviews)
which.max(reviews_moviefht$total_reviews)

# Summary 68534 users rating 9809 movies with a total number of ratings of
# 999999 avg rating 3.51 each user rated on average 14.6 movies. 
# On average each movie is rated 102 times. Pulp Fiction received the highest
# of ratings 3502


# Remove genre, title and time stamp from holdout

fht <- final_holdout_test %>% 
  select(-timestamp, - genres, -title)

# Coerce Final Holdout to a real rating matrix

fht_r <- as(fht, "realRatingMatrix")

# Reduce dimensions using the number of average number of movies rated by user
# and the average number of ratings received by movie

fht_r <- fht_r[rowCounts(fht_r) >= 15,
               colCounts(fht_r) >= 102]

# Run the same evaluation scheme on the final hold out

set.seed(76)
es_fht_r <-  evaluationScheme(fht_r, method = "cross-validation", given = 25,
                              goodRating = 4)

t_fht_r <- getData(es_fht_r, "train")
k_fht_r <- getData(es_fht_r, "known")
u_fht_r <- getData(es_fht_r, "unknown")

t_fht_r
k_fht_r
u_fht_r

# Training 8190 x 2015 rating matrix of class ‘realRatingMatrix’ with 
# 363528 ratings.
# Known - testing set 918 x 2015 rating matrix of class ‘realRatingMatrix’ 
# with 22983 ratings.
# Unknown - validation set 918 x 2015 rating matrix of class 
#‘realRatingMatrix’ with 17848 ratings.

# remove files for additional memory
rm(es_fht_r, fht, fht_r)

# use popular to test the parameter given
FHT_test <- Recommender(t_fht_r, method = "POPULAR")
FHT_eval_g <- FHT_test %>% 
  predict(k_fht_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_fht_r)

print(FHT_eval_g)

# Create recommender model on final test set using SVDF
set.seed(1304)
FHT_rec <- Recommender(t_fht_r, method = "SVDF")

# Evaluate recommneder model built using the final test set.
FHT_eval <- FHT_rec %>% 
  predict(k_fht_r, type = "ratings") %>% 
  calcPredictionAccuracy(u_fht_r)


# Compare SVDF models on the edx and final set
FHT_eval
SVDF_eval

test_eval <- data.frame(Model=c('FHT', 'SVDF'),
                         RMSE=c(FHT_eval[1], SVDF_eval[1]))
test_eval


te <-  test_eval %>% 
  ggplot(aes(reorder(Model,RMSE), RMSE)) +
  geom_bar(stat = 'identity', col = 'black', fill = 'red') +
  labs(x = 'Model' , y = 'RMSE', title = 'Model Accuracy')
te