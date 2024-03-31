import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# st.title("k-Nearest Neighbors (KNN) vLab Simulation")
# st.sidebar.image("https://vlabcomp.kjsieit.in/uploads/Somaiya1.png", caption="Computer Department-2024", use_column_width=True)
st.set_page_config(
    page_title="KNN VLab",
    page_icon="*",
)
# Streamlit UI
st.title("k-Nearest Neighbors (KNN) vLab Simulation")
st.sidebar.image("https://vlabcomp.kjsieit.in/uploads/Somaiya1.png", caption="Computer Department-2024", use_column_width=True)
st.sidebar.header("Simulation Settings")

k = st.sidebar.slider("Select the value of k", 1, 15, 3)

custom_dataset = st.sidebar.file_uploader("Upload a custom dataset (CSV)", type=["csv"])

if custom_dataset is not None:
    df = pd.read_csv(custom_dataset)    
    string_columns = df.select_dtypes(include=['object'])
    label_encoder = LabelEncoder()
    for col in string_columns:
        df[col] = label_encoder.fit_transform(df[col])

    df.to_csv('preprocessed_dataset.csv', index=False)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    st.write("Using custom dataset:")
    st.write(df)
    num_features = X.shape[1]
else:
    from sklearn import datasets
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['Class'] = iris.target
    X = iris.data
    y = iris.target
    st.write("Using default Iris dataset:")
    st.write(iris_df)
    num_features = X.shape[1]

if num_features > 2:
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

st.header("Dataset (2D Projection)")
st.write("Scatter plot of the dataset:")

fig, ax = plt.subplots()
for class_label in np.unique(y):
    data = X[y == class_label]
    ax.scatter(data[:, 0], data[:, 1], label=f"Class {class_label}")

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(title="Classes")

st.sidebar.header("Add a New Point")
new_x = st.sidebar.number_input("Enter X-coordinate", min_value=np.min(X[:, 0]), max_value=np.max(X[:, 0]))
new_y = st.sidebar.number_input("Enter Y-coordinate", min_value=np.min(X[:, 1]), max_value=np.max(X[:, 1]))

new_point = np.array([[new_x, new_y]])
predicted_class = knn.predict(new_point)[0]

colors = ["blue", "orange", "green"]  # Assume classes 0, 1, and 2 correspond to blue, green, and red
ax.scatter(new_x, new_y, marker='o', color=colors[predicted_class], label=f"New Point (Class {predicted_class})")

ax.annotate(f"Class {predicted_class}", (new_x, new_y), textcoords="offset points", xytext=(5, 5), ha='center')

st.pyplot(fig)
st.subheader('Model Performance')
st.write('ACCURACY')
accuracy = knn.score(X, y)
st.info(f"Accuracy for k = {k}: {accuracy * 100:.2f}%")
from sklearn.metrics import precision_score, recall_score, f1_score
st.write('PRECISION')
precision = precision_score(y, knn.predict(X), average='weighted')
st.info(f"Precision for k = {k}: {precision * 100:.2f}%")
st.write('RECALL')
recall = recall_score(y, knn.predict(X), average='weighted')
st.info(f"Recall for k = {k}: {recall * 100:.2f}%")
st.write('F1_SCORE')
f1 = f1_score(y, knn.predict(X), average='weighted')
st.info(f"F1 Score for k = {k}: {f1 * 100:.2f}%")
confusion = confusion_matrix(y, knn.predict(X))
st.subheader("Confusion Matrix")
st.write(confusion)

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
