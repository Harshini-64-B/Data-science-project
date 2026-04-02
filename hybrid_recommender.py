# hybrid_recommender.py
'''
Hybrid Recommendation System for user's Superstore dataset
- Loads uploaded CSVs from the paths
- Preprocesses data and creates features
- Trains classification models (LogisticRegression, RandomForest)
- Prints confusion matrix and metrics (accuracy, precision, recall, f1)
- Builds item-item cosine similarity and provides recommend_for_user() - gives the top 10 recommendations for a user based on combined ML model + CF similarity
'''

'''
Usage: "python3 hybrid_recommender.py" Run this command in terminal after placing the "Superstore-Dataset-Reviews.csv" and "Superstore-Data.csv" files in the same directory as this script. 
The script will print out the best model's performance metrics and an example recommendation for a random user. 
You can modify the parameters in the run_pipeline() function call to customize the behavior (e.g., change rating threshold, top_n recommendations, etc.).
'''

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def run_pipeline(
    ratings_path='Superstore-Dataset-Reviews.csv',
    products_path='Superstore-Data.csv',
    rating_threshold=4,
    test_size=0.2,
    random_state=42,
    top_n=10,
    combine_alpha=0.6
):

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    ratings = pd.read_csv(ratings_path, encoding='latin1')
    products = pd.read_csv(products_path, encoding='latin1')

    ratings = ratings.rename(columns=lambda x: x.strip())
    products = products.rename(columns=lambda x: x.strip())

    df = ratings[['Customer ID', 'Product ID', 'Product Name', 'Rate']].dropna()

    df['Customer ID'] = df['Customer ID'].astype(str)
    df['Product ID'] = df['Product ID'].astype(str)
    df['Product Name'] = df['Product Name'].astype(str)
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    df = df.dropna(subset=['Rate'])

    # Binary target: liked = 1 if rating >= threshold
    df['liked'] = (df['Rate'] >= rating_threshold).astype(int)

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    user_stats = df.groupby('Customer ID').Rate.agg(['mean', 'count'])
    user_stats.rename(columns={'mean': 'user_avg', 'count': 'user_count'}, inplace=True)

    item_stats = df.groupby('Product ID').Rate.agg(['mean', 'count'])
    item_stats.rename(columns={'mean': 'item_avg', 'count': 'item_count'}, inplace=True)

    df = df.merge(user_stats, left_on='Customer ID', right_index=True)
    df = df.merge(item_stats, left_on='Product ID', right_index=True)

    df['user_item_diff'] = df['user_avg'] - df['item_avg']

    features = ['user_avg', 'user_count', 'item_avg', 'item_count', 'user_item_diff']
    X = df[features]
    y = df['liked']

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---------------------------------------------------------
    # 3. Build Models
    # ---------------------------------------------------------
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    models = {
        'LogisticRegression': lr_pipeline,
        'RandomForest': rf
    }

    results = {}

    # ---------------------------------------------------------
    # 4. Train + Evaluate Models
    # ---------------------------------------------------------
    for name, model in models.items():
        print(f'\n--- Training {name} ---')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            'model': model,
            'confusion_matrix': cm,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Pick best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best = results[best_model_name]

    print("\nBest model by F1:", best_model_name)

    # ---------------------------------------------------------
    # 5. Confusion Matrix Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 4))
    sns.heatmap(best['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ---------------------------------------------------------
    # 6. Build Item–Item Similarity Matrix
    # ---------------------------------------------------------
    user_item = df.pivot_table(
        index='Customer ID',
        columns='Product ID',
        values='Rate',
        aggfunc='mean',
        fill_value=0
    )

    item_matrix = user_item.T
    item_sim = cosine_similarity(item_matrix)
    item_sim_df = pd.DataFrame(item_sim, index=item_matrix.index, columns=item_matrix.index)

    # ---------------------------------------------------------
    # 7. Recommendation Function
    # ---------------------------------------------------------
    def recommend_for_user(user_id, top_n=top_n, combine_alpha=combine_alpha, model_name=best_model_name):

        if user_id not in user_item.index:
            raise ValueError("Unknown user_id")

        user_ratings = user_item.loc[user_id]
        unseen_items = user_ratings[user_ratings == 0].index.tolist()

        if len(unseen_items) == 0:
            return pd.DataFrame()

        # User-level features
        if user_id in user_stats.index:
            u_avg = user_stats.loc[user_id, 'user_avg']
            u_count = user_stats.loc[user_id, 'user_count']
        else:
            u_avg = df['Rate'].mean()
            u_count = 0

        candidates = []

        for item in unseen_items:
            if item in item_stats.index:
                i_avg = item_stats.loc[item, 'item_avg']
                i_count = item_stats.loc[item, 'item_count']
            else:
                i_avg = df['Rate'].mean()
                i_count = 0

            candidates.append({
                'Product ID': item,
                'user_avg': u_avg,
                'user_count': u_count,
                'item_avg': i_avg,
                'item_count': i_count,
                'user_item_diff': u_avg - i_avg
            })

        cand_df = pd.DataFrame(candidates)
        X_cand = cand_df[features]

        clf = results[model_name]['model']

        if hasattr(clf, 'predict_proba'):
            cand_df['clf_prob'] = clf.predict_proba(X_cand)[:, 1]
        else:
            scores = clf.decision_function(X_cand)
            cand_df['clf_prob'] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        # Similarity score
        liked_items = df[(df['Customer ID'] == user_id) & (df['Rate'] >= rating_threshold)]['Product ID'].unique()

        sim_scores = []
        for item in cand_df['Product ID']:
            existing = [li for li in liked_items if li in item_sim_df.columns]
            if len(existing) == 0:
                sim_scores.append(0.0)
            else:
                sims = item_sim_df.loc[item, existing].values
                sim_scores.append(np.mean(sims))

        cand_df['sim_score'] = sim_scores

        # Combine CF + ML model
        cand_df['combined_score'] = combine_alpha * cand_df['clf_prob'] + (1 - combine_alpha) * cand_df['sim_score']

        prod_names = df[['Product ID', 'Product Name']].drop_duplicates().set_index('Product ID')['Product Name'].to_dict()
        cand_df['Product Name'] = cand_df['Product ID'].map(prod_names)

        top = cand_df.sort_values('combined_score', ascending=False).head(top_n)
        return top[['Product ID', 'Product Name', 'clf_prob', 'sim_score', 'combined_score']]

    # ---------------------------------------------------------
    # 8. Example Recommendation
    # ---------------------------------------------------------
    example_user = df['Customer ID'].sample(1, random_state=random_state).iloc[0]
    print("\nExample User:", example_user)
    print(recommend_for_user(example_user))

    # ---------------------------------------------------------
    # 9. Print Best Model Metrics
    # ---------------------------------------------------------
    print("\nBest Model Metrics:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1 Score: {best['f1']:.4f}")

    # ---------------------------------------------------------
    # 10. Explainability: Feature Importances
    # ---------------------------------------------------------
    print("\nExplainability:")
    if best_model_name == "LogisticRegression":
        coef = best['model'].named_steps['clf'].coef_[0]
        for f, c in zip(features, coef):
            print(f"{f}: {c:.4f}")

    elif best_model_name == "RandomForest":
        importances = best['model'].feature_importances_
        for f, imp in zip(features, importances):
            print(f"{f}: {imp:.4f}")

    return results, recommend_for_user


if __name__ == '__main__':
    run_pipeline()