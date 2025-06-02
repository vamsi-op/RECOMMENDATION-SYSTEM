# RECOMMENDATION-SYSTEM

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: VAMSI PUTTEPU

*INTERN ID*: CT04DN135

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## 🧠 Task 4: Movie Recommendation System using SVD – Detailed Description

For the final task in the CODTECH Machine Learning Internship, we built a personalized **Recommendation System** using the collaborative filtering approach with **Singular Value Decomposition (SVD)**. The objective was to recommend movies to users based on past interactions, simulating the core functionality behind platforms like Netflix and Amazon.

To accomplish this, we used the **MovieLens 100k dataset**, which contains 100,000 movie ratings from 943 users across 1682 movies. It is a gold standard dataset in the recommendation system domain and includes user-item interaction data in the form of ratings ranging from 1 to 5.

---

### 📁 Dataset and Tools

The dataset was accessed using the `surprise` library’s built-in loader for `ml-100k`. Surprise, short for **Simple Python RecommendatIon System Engine**, is designed specifically for collaborative filtering models and offers various matrix factorization and KNN-based algorithms.

We encountered a technical challenge due to compatibility issues between `numpy` version 2.x and `scikit-surprise`. This was resolved by downgrading NumPy to version `1.23.5` and installing `scikit-surprise` from source using `--no-binary`. These adjustments were essential to successfully build and train our model.

---

### 🧠 Model Architecture: SVD

**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes the user-item rating matrix into lower-dimensional representations. These latent factors are then used to predict missing entries — in this case, ratings not yet given by users to certain movies.

---

### 🔧 Implementation Steps

1. **Data Preparation**: Loaded MovieLens 100k using `Dataset.load_builtin('ml-100k')`.
2. **Splitting**: Divided the dataset into an 80/20 train-test split using Surprise's utility functions.
3. **Model Training**: Trained the `SVD` model on the training set, which learned the hidden relationships between users and items.
4. **Prediction**: Made predictions on the test set and compared them with actual user ratings.
5. **Evaluation**:
   - Used **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** as evaluation metrics.
   - Applied **5-fold cross-validation** to ensure generalization and prevent overfitting.
6. **Generating Recommendations**: Implemented a function `get_top_n()` that generated the top-N movie predictions per user by sorting their predicted ratings.

---

### 📊 Results

The trained model delivered reasonably accurate predictions with low RMSE and MAE scores. It was able to learn from historical ratings and estimate how a user would rate new movies. For example, we printed the top 5 movie IDs with the highest predicted ratings for a specific user (user ID `196`), demonstrating real-world applicability.

---

### 🔍 Insights

This task emphasized the power of collaborative filtering in making personalized predictions without needing explicit item metadata (like genres or tags). The model’s ability to infer user preferences from only interaction data is what powers many industrial-grade recommendation engines today.

One of the highlights of the task was learning to handle environment-specific errors such as binary compatibility. This added practical value in deploying ML code across different systems.

---

### ✅ Conclusion

By completing this task, we’ve gained hands-on experience with one of the most foundational algorithms in recommender systems. SVD is an effective technique that remains relevant in production environments due to its simplicity and performance.

We explored not just the algorithmic theory but also implementation, evaluation, and tuning — covering a full development cycle for a machine learning-based recommendation engine. This project completes our internship with a comprehensive understanding of supervised learning, NLP, deep learning, and recommender systems — all critical pillars in the machine learning landscape.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/489f8769-a8bf-45ce-ab87-9ae06826a515)
