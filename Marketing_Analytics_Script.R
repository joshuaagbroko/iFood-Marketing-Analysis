---
  title: "**iFood Marketing Analysis**"
author: "**Agbroko Joshua**"
output:
  html_document: default
pdf_document: default
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## **Project Scenario**
I'm a marketing data analyst and I've been told by the Chief Marketing Officer of the fictional iFood Company that recent marketing campaigns have not been as effective as they were expected to be. I need to analyze the data set to understand this problem and propose data-driven solutions. In this dataset’s Kaggle page, there are some EDA directions that the data publisher suggested following and I decided to choose some to explore.

## **Dataset Overview**
The dataset for this project is provided by Dr. Omar Romero-Hernandez. It is licensed as CC0: Public Domain, which states, “You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission” You can also see the license status and download this dataset on this [Kaggle page](https://www.kaggle.com/jackdaoud/marketing-data).



##      **Table of Contents**
### Section 01: Exploratory Data Analysis  
* Handling missing values
* Feature engineering
* Do you notice any patterns or anomalies in the data ? Can these be plotted ?  
  
  ### Section 02: Statistical Analysis
  * What factors are significantly related to the number of store purchases?
  * My supervisor insists that people who buy gold are more conservative. justify or refute this statement using an appropriate statistical test.
* Fish has Omega 3 fatty acids which are good for the brain. Accordingly, do "Married PhD candidates" have a significant relation with amount spent on fish?  
  
  ### Section 03: Data Visualization  
  * Which marketing campaign is more successful
* What does the average customer look like for this company?
  * Which products are performing best and Which channels are underperforming

### Section 04: Formulating Data-Driven Solutions  
* Bringing together everything to provide data_driven recommendations for my CMO.  



Before we dive we need to load the dataset and all required packages for this project.

```{r message=FALSE, warning=FALSE}
# load packages
library(tidyverse)
library(readxl)
library(dplyr)
library(cowplot)
library(corrplot)
library(shapley)
library(shapper)
library(ggplot2)
library(stats)
library(reshape2)
library(randomForest)
library(caret)


df <- read_excel("~/R/Pathway Projects/portfolio/Marketing_Analytics-main/marketing_data.xlsx")


### Summary
head(df)

basic_info <- function(df) {
  cat("This dataset has", ncol(df), "columns and", nrow(df), "rows.\n")
  cat("This dataset has", nrow(df[duplicated(df), ]), "duplicated rows.\n\n")
  cat("Descriptive statistics of the numeric features in the dataset:\n\n")
  print(summary(df))
  cat("\nInformation about this dataset:\n\n")
  print(str(df))
}

basic_info(df)


### Handling missing values
df <- df %>%
  mutate(Income = ifelse(is.na(Income), 0, Income))


### Section 1.1: Checking for outliers
# Selecting columns to plot
df_to_plot <- df %>%
  select(-c(ID, AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Response, Complain)) %>%
  select_if(is.numeric)

# Defining the number of rows and columns for subplots
num_cols <- 4
num_rows <- ceiling(ncol(df_to_plot) / num_cols)

# Creating a list of boxplot objects
boxplots <- lapply(names(df_to_plot), function(col_name) {
  ggplot(df_to_plot, aes_string(y = col_name)) +
    geom_boxplot(fill = "lightblue") +
    labs(y = col_name) +
    theme_minimal()
})

# Arranging  the boxplots in a grid layout
plot_grid(plotlist = boxplots, ncol = num_cols)



### Section 1.2: Feature Engineering
# Creating new features
new_df <- df %>%
  mutate(Join_year = lubridate::year(Dt_Customer),
         Join_month = lubridate::month(Dt_Customer),
         Join_weekday = lubridate::wday(Dt_Customer),
         Minorhome = Kidhome + Teenhome,
         Total_Mnt = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds,
         Total_num_purchase = NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases + NumWebVisitsMonth,
         Total_accept = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp2 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 + Response,
         AOV = Total_Mnt / Total_num_purchase)


# Displaying the first 6 rows 
head(new_df)


# Converting Dt_Customer to date format
new_df$Dt_Customer <- as.Date(new_df$Dt_Customer)

## Section 1.3: Do you notice any patterns or anomalies in the data ? Can these be plotted ?  

# Selecting numeric columns to plot
df_to_plot <- new_df[, sapply(new_df, is.numeric)]
# Remove the 'ID' column if it exists
df_to_plot <- df_to_plot[, !(names(df_to_plot) %in% c("ID"))]

# Compute correlation matrix
correlation_matrix <- cor(df_to_plot)

# Convert correlation matrix to long format
cor_df <- melt(correlation_matrix)

# Create heatmap plot
ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0,
                       breaks = seq(-1, 1, by = 0.2)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap", x = NULL, y = NULL)


### Anomaly:
# Visualizing NumWebPurchases vs NumWebVisitsMonth 
plot(new_df$Complain, new_df$Total_Mnt, 
     xlab = "Complain", ylab = "Total_Mnt", 
     main = "Complain vs Total_Mnt", 
     col = "blue", pch = 19)


# *Section 02: Statistical Analysis*
# Create a histogram of NumStorePurchases
hist(new_df$NumStorePurchases, 
     main = "Distribution of the number of store purchases",
     xlab = "Number of Store Purchases",
     ylab = "Frequency",
     col = "skyblue",
     border = "black")


## Section 2-1: What factors are significantly related to the number of store purchases?

# Drop ID and Dt_Customer columns
rd_df <- new_df[, !(names(new_df) %in% c("ID", "Dt_Customer"))]


# One-hot encoding
rd_df <- dummyVars("~.", data = rd_df) %>%
  predict(rd_df)
# Split dataset into features (X) and labels (y)
X <- subset(rd_df, select = -NumStorePurchases)
y <- rd_df[, "NumStorePurchases", drop = FALSE]



# Split dataset into training set and test set
set.seed(123) # for reproducibility
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Replace infinite values with zero in X_train
X_train[is.infinite(X_train)] <- 0

# Create a Random Forest model
rg <- randomForest(X_train, y_train, ntree = 200)

# Predict on the test set
y_pred <- predict(rg, X_test)

# Calculate evaluation metrics
mae <- mean(abs(y_test - y_pred))
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mean((y_test - y_pred)^2))

# Print evaluation metrics
cat("Mean Absolute Error:", mae, "\n")
cat("Mean Squared Error:", mse, "\n")
cat("Root Mean Squared Error:", rmse, "\n")

# Get feature importance scores
feature_imp <- importance(rg)
feature_imp <- as.data.frame(feature_imp)
feature_imp$Feature <- rownames(feature_imp)
colnames(feature_imp) <- c("Importance", "Feature")
feature_imp <- feature_imp %>% arrange(desc(Importance))

# Select top 10 important features
top_10_features <- head(feature_imp, 10)

# Create bar plot
ggplot(top_10_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(x = "Features", y = "Feature Importance Score", title = "Top 10 Important Features") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10)) +  # Adjust font size
  theme(axis.title.y = element_text(size = 12, face = "bold")) +  # Adjust font size and style
  theme(axis.title.x = element_text(size = 12, face = "bold")) +  # Adjust font size and style
  theme(plot.title = element_text(size = 14, face = "bold")) +  # Adjust font size and style
  geom_text(aes(label = round(Importance, 2)), hjust = -0.3, size = 3)  # Add labels to bars

# Filtering store shoppers
store_shoppers <- subset(new_df, NumStorePurchases > 0)
store_shoppers <- subset(store_shoppers, AOV <= (mean(store_shoppers$AOV) + 3 * sd(store_shoppers$AOV)))
store_shoppers$`Type of shopper` <- "In-store"

# Filtering other shoppers
other_shoppers <- subset(new_df, NumStorePurchases == 0)
other_shoppers$`Type of shopper` <- "Other"

# Combining both types of shoppers
all_shoppers <- rbind(store_shoppers, other_shoppers)

# Creating boxplot
ggplot(all_shoppers, aes(x = `Type of shopper`, y = AOV, fill = `Type of shopper`)) +
  geom_boxplot() +
  labs(title = "Do in-store shoppers have a higher average order volume?",
       x = "Type of Shopper",
       y = "Average Order Volume") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        legend.position = "none") +
  scale_fill_manual(values = c("In-store" = "skyblue", "Other" = "lightgreen")) +
  theme(legend.title = element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))  

# Calculate average amount spent on gold
average_gold_spent <- mean(new_df$MntGoldProds)

# Create a scatter plot
plot(new_df$MntGoldProds, new_df$NumStorePurchases,
     xlab = "Amount Spent on Gold (Last 2 Years)",
     ylab = "Number of Store Purchases",
     main = "Scatter Plot: Gold Spending vs Store Purchases")

# Add a horizontal line at the average gold spending
abline(h = average_gold_spent, col = "red", lty = 2)

# Add a legend
legend("topright", legend = c("Average Gold Spending"), col = c("red"), lty = 2, cex = 0.8)


# Calculate correlation coefficient
correlation <- cor(new_df$MntGoldProds, new_df$NumStorePurchases)
cat("Correlation coefficient:", correlation, "\n")

# Perform correlation test
correlation_test <- cor.test(new_df$MntGoldProds, new_df$NumStorePurchases)

# Print the results
print(correlation_test)

## Section 2.3. Fish has Omega 3 fatty acids which are good for the brain. Accordingly, do “Married PhD candidates” have a significant relation with amount spent on fish?


# Divide the data into two groups: married PhD candidates and the rest
married_phd <- new_df[new_df$Marital_Status == "Married" & new_df$Education == "PhD", "MntFishProducts"]
rest <- new_df[!(new_df$Marital_Status == "Married" & new_df$Education == "PhD"), "MntFishProducts"]

# Extract numeric vectors from tibbles
married_phd <- married_phd$MntFishProducts
rest <- rest$MntFishProducts


# Creating a boxplot to visualize the distribution of amount spent on fish for the two groups
boxplot(married_phd, rest,
        names = c("Married PhD", "Rest"),
        xlab = "Group",
        ylab = "Amount Spent on Fish",
        main = "Amount Spent on Fish by Marital Status and Education",
        col = c("skyblue", "lightgreen"))

# Performing t-test
t_test_result <- t.test(married_phd, rest)
print(t_test_result)


## *Section 03: Data Visualization and Further Analysis*


### 3.1 Which marketing campaign is most succesful ?
# Calculating  the sum of each marketing campaign
campaign_sum <- colSums(new_df[c("AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response")])

# Sorting the values in ascending order
campaign_sum <- sort(campaign_sum, decreasing =  FALSE)

par(mar = c(7, 10, 4, 2) + 0.1) 

# Creating a horizontal bar plot
barplot(campaign_sum,
        horiz = TRUE, # Horizontal bar plot
        main = "Which marketing campaign is most successful?",
        xlab = "Offer Accepted",
        
        col = "skyblue", 
        las = 1, 
        xlim = c(0, max(campaign_sum) * 1.1), 
        cex.names = 0.8) 

### 3.2 What does the average customer look like for this company? Which products are performing best?

# Replace infinite values with zero
new_df <- new_df %>%
  mutate_all(~ifelse(is.infinite(.), 0, .))

# Excluding non-numeric columns
numeric_df <- new_df[, sapply(new_df, is.numeric)]

# Calculating the mean of all numeric items in the dataset
mean_values <- colMeans(numeric_df)


# Format the mean values for better readability
formatted_mean_values <- format(mean_values, scientific = FALSE)

# Displaying the mean values
cat("This is what an average customer looks like:\n")
print(formatted_mean_values)

### 3.3 Performance of Marketing Channels 
# Select the columns of interest and calculate the sum
purchase_sum <- colSums(new_df[, c("NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases")])

# Sorting the values
purchase_sum <- purchase_sum[order(purchase_sum)]

# Creating the horizontal bar plot

par(mar = c(6, 12, 4, 5) + 0.1)

barplot(purchase_sum,
        horiz = TRUE, 
        main = "Which marketing channels are underperforming?",
        xlab = "Total number of purchases",
      
        col = "skyblue", 
        las = 1,
        ) 


## *3.4 Further Investigation*

# Creating two groups based on the acceptance of offers
cp_last <- new_df[new_df$Response > 0, ]
cp_the_rest <- new_df[new_df$AcceptedCmp2 == 0, ]

# Getting the number of observations in each group
num_cp_last <- nrow(cp_last)
num_cp_the_rest <- nrow(cp_the_rest)

# Displaying the number of observations in each group
cat("Number of observations in the group that accepted offers from the last campaign:", num_cp_last, "\n")
cat("Number of observations in the group that did not accept offers from the last campaign:", num_cp_the_rest, "\n")

# Creating a copy of cp__the_rest
cp_the_rest2 <- cp_the_rest

# Find overlapping IDs and remove them from cp_the_rest2
overlap_ids <- intersect(cp_the_rest$ID, cp_last$ID)
cp_the_rest2 <- cp_the_rest2[!cp_the_rest2$ID %in% overlap_ids, ]

# Getting the number of observations in each group after removal of overlaps
num_cp_last <- nrow(cp_last)
num_cp_the_rest <- nrow(cp_the_rest2)

# Displaying the number of observations in each group after removal of overlaps
cat("Number of customers in the group that accepted offers from the last campaign:", num_cp_last, "\n")
cat("Number of customers in the group that did not accept offers from the last campaign after removal of overlaps:", num_cp_the_rest, "\n")

# Select the desired columns for cp_last
cp_last_selected <- cp_last[, c('Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                                'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                                'Total_Mnt', 'Total_num_purchase', 'AOV')]



# Calculate the mean of the selected columns
cp_last_mean <- colMeans(cp_last_selected, na.rm = TRUE)

# Format the mean values for better readability
formatted_cp_last_mean <- format(cp_last_mean, scientific = FALSE)

# Displaying the mean values
cat("This is what an average customer from the last campaign looks like:\n")

print(formatted_cp_last_mean)

# Select the desired columns for cp_last
cp_the_rest_selected <- cp_the_rest2[, c('Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                                'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                                'Total_Mnt', 'Total_num_purchase', 'AOV')]



# Calculate the mean of the selected columns
cp_the_rest_mean <- colMeans(cp_the_rest_selected, na.rm = TRUE)

# Format the mean values for better readability
formatted_cp_the_rest_mean <- format(cp_the_rest_mean, scientific = FALSE)

# Displaying the mean values
cat("This is what an average customer from the campaign 1-5 looks like:\n")

print(formatted_cp_the_rest_mean)

# Select the desired columns for cp_last
new_df2_selected <- new_df[, c('Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                                'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                                'Total_Mnt', 'Total_num_purchase', 'AOV')]



# Calculate the mean of the selected columns
new_df2_mean <- colMeans(new_df2_selected, na.rm = TRUE)

# Format the mean values for better readability
formatted_new_df2_mean <- format(new_df2_mean, scientific = FALSE)

# Displaying the mean values
cat("This is what an average customer looks like:\n")

print(formatted_new_df2_mean)

# Calculate the percentage differences
percentage_diff <- ((colMeans(cp_last_selected, na.rm = TRUE) - colMeans(new_df2_selected, na.rm = TRUE)) / colMeans(new_df2_selected, na.rm = TRUE)) * 100

# Remove NA values
percentage_diff <- percentage_diff[!is.na(percentage_diff)]

# Sort the values
percentage_diff_sorted <- sort(percentage_diff)

# Determine colors
colors <- ifelse(percentage_diff_sorted >= 0, "navy", "orange")

# Create a bar plot
par(mar = c(7, 10, 4, 2)) #margins (bottom, left, top, right)

barplot(percentage_diff_sorted, horiz = TRUE, col = colors, main = "Customer Characteristics Comparison - Customer in last campaign vs Average customer",
      xlab = "Difference in %", names.arg = names(percentage_diff_sorted), las = 1) 

# Calculate the percentage differences
percentage_diff2 <- ((colMeans(cp_last_selected, na.rm = TRUE) - colMeans(cp_the_rest_selected, na.rm = TRUE)) / colMeans(cp_the_rest_selected, na.rm = TRUE)) * 100

# Remove NA values
percentage_diff2 <- percentage_diff2[!is.na(percentage_diff2)]

# Sort the values
percentage_diff_sorted2 <- sort(percentage_diff2)

# Determine colors
colors <- ifelse(percentage_diff_sorted2 >= 0, "navy", "orange")

# Create a bar plot
par(mar = c(7, 10, 4, 2)) #margins (bottom, left, top, right)

barplot(percentage_diff_sorted2, horiz = TRUE, col = colors, main = "Customer Characteristics Comparison - Customer in last campaign vs Customers in Campaign 1-5",
      xlab = "Difference in %",  names.arg = names(percentage_diff_sorted), las = 1) 


# Counting the occurrences of each country
country_counts <- table(new_df$Country)

# Displaying the counts
print(country_counts)


cp_last_country <- cp_last %>%
  group_by(Country) %>%
  summarise(Percent = n() / nrow(cp_last) * 100) %>%
  arrange(Country)

# Print the resulting dataframe
print(cp_last_country)


cp_the_rest2_country <- cp_the_rest2 %>%
  group_by(Country) %>%
  summarise(Percent = n() / nrow(cp_the_rest2) * 100) %>%
  arrange(Country)

# Print the resulting dataframe
print(cp_the_rest2_country)

# Calculate the difference in percentages between cp_last_country and cp__the_rest2_country
country_final <- merge(cp_last_country, cp_the_rest2_country, by = "Country", suffixes = c("_cp_last", "_cp_rest"))
country_final$Difference <- country_final$Percent_cp_last - country_final$Percent_cp_rest

# Visualize the differences
ggplot(country_final, aes(x = reorder(Country, Difference), y = Difference, fill = Difference > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("navy", "orange"), labels = c("Positive", "Negative")) +
  labs(title = "Country Percent Comparison - The last campaign vs Campaign 1-5",
       x = "Country",
       y = "Difference in %") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = FALSE) +
  coord_flip()


# List of columns
list <- c('MntWines', 'MntMeatProducts', 'MntGoldProds', 'MntFishProducts', 'MntFruits', 'MntSweetProducts')

# Loop through each column
for (i in list) {
  # Calculate Pearson correlation coefficient and p-value
  result <- cor.test(x = new_df[[i]], y = new_df$Total_accept, method = "pearson")
  
  # Print results
  cat(i, "vs Total_accept:\n")
  cat('Pearson correlation (r): ', result$estimate, '\n')
  cat('Pearson p-value: ', result$p.value, '\n\n')
}
