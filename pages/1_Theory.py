import streamlit as st
st.title("k-Nearest Neighbors (KNN) vLab")
st.markdown(
    """
    The k-Nearest Neighbors (KNN) algorithm is a simple yet effective machine learning classification and regression method. It's based on the idea that similar data points are close to each other in a feature space. Here's a brief theoretical overview of the KNN algorithm:

**KNN Algorithm Overview:**

1. **Training Phase:**
   - In the training phase, the algorithm simply stores the feature vectors and their corresponding class labels from the training dataset.

2. **Prediction Phase:**
   - When a new, unlabeled data point is given for prediction, KNN identifies the K-nearest data points in the training dataset based on a similarity metric (typically Euclidean distance).
   - The value of K is a user-defined hyperparameter and determines the number of neighbors to consider.
   - The class of the new data point is predicted based on the majority class among its K-nearest neighbors.
   - In the case of regression, the predicted value can be calculated as the mean or median of the target values of the K-nearest neighbors.

3. **Choosing K:**
   - The choice of K is critical. A smaller K may make the model sensitive to noise, while a larger K may lead to overly smooth boundaries.
   - It's common to use cross-validation to select an optimal value for K.

**Key Considerations and Concepts:**

- **Distance Metric:** The choice of distance metric is essential. Euclidean distance is commonly used, but other metrics like Manhattan, Minkowski, or custom distances can be employed based on the nature of the data.

- **Weighted KNN:** In some variations, you can give more weight to the closer neighbors, meaning that the neighbors' influence on the prediction is inversely proportional to their distance from the new data point.

- **Scaling Features:** Feature scaling is crucial because KNN is sensitive to the scale of features. Normalizing or standardizing features can improve the algorithm's performance.

- **Curse of Dimensionality:** KNN can be affected by the curse of dimensionality. As the number of features increases, the distance between data points tends to become similar, which can impact the algorithm's performance.

- **Handling Ties:** In classification problems, it's possible to have a tie among the classes of the K-nearest neighbors. Several strategies can be used to handle ties, such as selecting the class with the most frequent neighbors or using weighted voting.

**Strengths:**
- Simple and easy to understand.
- Non-parametric (it makes no assumptions about the data distribution).
- Effective for both classification and regression problems.
- Robust to noisy data and outliers.

**Weaknesses:**
- Computationally intensive, especially for large datasets, as it requires calculating distances for all data points.
- Sensitive to the choice of K and the distance metric.
- Doesn't provide insights into the importance of features.

**Applications:**
- KNN is used in a wide range of applications, including recommendation systems, image recognition, natural language processing, and more.

**Summary:**
KNN is a versatile and intuitive algorithm that can be a good choice for simple classification and regression tasks. However, its performance may be influenced by the choice of K, the distance metric, and the nature of the data, so it's important to experiment with different settings and validate the model's performance on real-world data.
    """
)
st.markdown(
    """
    ***
    <div class="footer" style='text-align:center; color:#9b2928;font-weight: bold;'>
    Department Of Computer Engineering\n
    <b>Guided by: Prof.Kavita Bathe\n
    Developed by: Arvind Patel, Dev Shah, Avisha Shah</b>
    </div>
    """,
    unsafe_allow_html=True
)
