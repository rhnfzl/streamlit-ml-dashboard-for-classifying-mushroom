import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App") #adding the header to the sidebar as well as the sidebar
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache(persist=True) #to cache the data
    #load the csv file using pandas and label encoding sklearn LabelEncoder
    def load_data():
        data = pd.read_csv('/mushrooms.csv')
        label = LabelEncoder()
        #each column name from pandas
        for col in data.columns:
            #for each row in the data label encode them
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True) 
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Preciosn-Recall Curve' in metrics_list: #typically used to measure the performance of binary classifier
            st.subheader("Preciosn-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous'] #defining class for one hot encoding
    st.sidebar.subheader("Choose Classifier")
    #dropdown list and the options are in the tuples of string
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"), )

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters") #asking user to provide model hyperprameters
        #relularization paramter; adding a box to input the number
        C = st.sidebar.number_input("C (Regularization Prameter)", 0.01, 10.0, step=0.01, key='C')
        #adding radio button for kernels
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        #adding gamma which is kernel cofficents in radio format
        gamma = st.sidebar.radio("Gamma Kernel Coefficient", ("scale", "auto"), key='gamma')

        #adding widget to the sidebar
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix","ROC Curve", "Preciosn-Recall Curve"))

        #adding classify button so that it shouldn't update with every parameter tuning
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test) #calculate the accuracy
            y_pred = model.predict(x_test) #calculate the prediction
            st.write("Accuracy : ", accuracy.round(2)) #write accuracy on screen
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names)) #write precision on screen
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2)) #write recall on screen
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters") #asking user to provide model hyperprameters
        #relularization paramter; adding a box to input the number
        C = st.sidebar.number_input("C (Regularization Prameter)", 0.01, 10.0, step=0.01, key='C_lr')
        #adding slider widgets
        max_iter = st.sidebar.slider("mMaximum number of iteration", 100, 500, key='max_iter')

        #adding widget to the sidebar
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix","ROC Curve", "Preciosn-Recall Curve"))

        #adding classify button so that it shouldn't update with every parameter tuning
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test) #calculate the accuracy
            y_pred = model.predict(x_test) #calculate the prediction
            st.write("Accuracy : ", accuracy.round(2)) #write accuracy on screen
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names)) #write precision on screen
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2)) #write recall on screen
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters") #asking user to provide model hyperprameters
        #input number widget
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The max depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        #adding widget to the sidebar
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix","ROC Curve", "Preciosn-Recall Curve"))

        #adding classify button so that it shouldn't update with every parameter tuning
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test) #calculate the accuracy
            y_pred = model.predict(x_test) #calculate the prediction
            st.write("Accuracy : ", accuracy.round(2)) #write accuracy on screen
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names)) #write precision on screen
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2)) #write recall on screen
            plot_metrics(metrics)

    #Checkbox to show data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df) #write the data frame to webapp
    
if __name__ == '__main__':
    main()
