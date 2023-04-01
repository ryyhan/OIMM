import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA


st.title("OIMM (One Interface Multiple Models) - Toy Datasets")


dataset_name = st.sidebar.selectbox(
    "Select a Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "Titanic Dataset")
)
st.write("The dataset selected is ", dataset_name, "dataset")

model_name = st.sidebar.selectbox("Select a Model", ("KNN", "SVM", "Random Forest"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write("Shape of Dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter(clf_name):
    params = dict()

    if clf_name == "KNN":
        neighbors = st.sidebar.slider("neighbors", 1, 15)
        params["neighbors"] = neighbors

    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter(model_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["neighbors"])
    elif clf_name == "SVM":
        model = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        model = RandomForestClassifier(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            random_state=42,
        )
    return model


model = get_classifier(model_name, params)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

st.write("Training set score =", model.score(X_train, y_train))
st.write("Test set score=", model.score(X_test, y_test))

# Visualization
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, cmap="viridis")
plt.colorbar()
st.set_option("deprecation.showPyplotGlobalUse", False)
st.pyplot()
