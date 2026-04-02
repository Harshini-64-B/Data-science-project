# Hybrid Recommendation System (Superstore Dataset)

## Overview
This project implements a hybrid recommendation system using the Superstore dataset. 
The goal is to recommend products to users by combining machine learning techniques 
and collaborative filtering.

The system predicts whether a user will like a product and also considers similarity 
between products to improve recommendations.


## Dataset

Two datasets are used in this project:

1. Superstore-Data.csv
   - Contains product and transaction details
   - Includes fields like Customer ID, Product ID, Product Name, Sales, etc.

2. Superstore-Dataset-Reviews.csv
   - Contains user ratings for products
   - Columns used:
     - Customer ID
     - Product ID
     - Product Name
     - Rate


## Approach

### 1. Data Preprocessing
- Removed missing values
- Converted data types properly
- Created a binary target:
  - liked = 1 if rating ≥ 4
  - liked = 0 otherwise

### 2. Feature Engineering
Additional features created:
- User average rating -> Gives the number of ratings by user
- Item average rating -> Gives the number of ratings for item
- Difference between user and item rating

### 3. Machine Learning Models
- Logistic Regression
- Random Forest Classifier

These models predict the probability that a user will like a product.

### 4. Collaborative Filtering
- Item–item similarity using cosine similarity
- Based on user-item interaction matrix
- Helps find similar products

### 5. Hybrid Recommendation
Final score:

combined_score = α * ML_score + (1 - α) * similarity_score


## Evaluation

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Best model is selected based on F1 score.


## Output

- Top N recommended products for a user (N is fixed as 10 here, can be changed)
- Model performance metrics
- Confusion matrix visualization
- Feature importance


## How to Run

1. Place datasets in the same folder as the code file:
   - Superstore-Data.csv
   - Superstore-Dataset-Reviews.csv
   - hybrid_recommender.py

2. In the command prompt or VS code, change the directory to the project directory.

3. Run:
   python hybrid_recommender.py


## Notes

- Combines classification and collaborative filtering
- Improves recommendation quality
- Works even with limited data

