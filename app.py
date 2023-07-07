##################### Import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import streamlit as st # pip install streamlit
from streamlit_lottie import st_lottie # pip install streamlit-lottie
from streamlit_option_menu import option_menu as om 
import pickle
#############

st.set_page_config(layout = 'wide')
st.set_option('deprecation.showPyplotGlobalUse', False)


#=============Function to filter DataFrame=======================

@st.cache_resource(experimental_allow_widgets=True)
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")
    if not modify:
        return df
    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df

#===========================Import Raw Data===========================
data_url = "train.csv"
df = pd.read_csv(data_url, index_col= 0)
#===========================Layout===========================
# Sidebar (Home, EDA, Prediction, More visualization)
with st.sidebar:
    selected =  om("Navigation", ['Home', 'EDA', 'Prediction', 'More Visualization'],
                    icons = ['house', 'flag', 'airplane', 'android'], 
                    menu_icon=  'list-task', default_index= 1, orientation= 'vertical')
##########################################################
#===================================Home===================================

if selected  == 'Home':
    st_lottie("https://assets7.lottiefiles.com/packages/lf20_3vbOcw.json")
    st.write("## Welcome to my demo !")
    st.title("The Survival of Titanic Passengers")
    st.write(""" The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. 
                Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out 
                of 2224 passengers and crew.""" )

    st.write("This project used some basic machine learning algorithms to predict whether a passenger would survive or not.")

#=======================EDA========================

if selected == 'EDA':
   # Title 
    st.title(":red[Exploratory Data Analysis]")
    st.write('**Please select the checkboxes that you want to explore on the side panel.**')
    #####First look include Basic Information , Filtering, Visualization: 
    st.sidebar.header("Quick Exploration")    
    #==================Basic Information=============
    if st.sidebar.checkbox('**Basic Information**'):
        # Basic Information
        if st.checkbox(':blue[Data Information]'):
            image_1 = Image.open('Data Dictionary.png')
            st.image(image_1)
            image_2 = Image.open('Variable notes.png')
            st.image(image_2)
        if st.checkbox(":blue[Show Raw Data]", False):
           st.subheader('Raw data')
           st.write(df)
        if st.sidebar.checkbox('Quick look'):
            st.divider()
            st.header("Let's see 10 first observations")
            st.write(df.head(10))
        if st.sidebar.checkbox('Descriptive Statistic'):
            st.divider()
            st.header("Descriptive Statistic")
            st.write(df.describe())
        if st.sidebar.checkbox('Check missing values'):
            st.divider()
            st.header("Missing values")
            st.write(df.isnull().sum())
            st.subheader('Plotting missing values of each column')
            st.bar_chart(data = df.isnull().sum())
            st.write('**There are 891 observations, but Cabin has 687 missing values.**')
    #==================Filtering=============
    if st.sidebar.checkbox('**Filtering**'):
        st.divider()
        st.write('**You can do some interesting filters here.**')
        st.dataframe(filter_dataframe(df), hide_index= True)
    #==================Visualization=============
    if st.sidebar.checkbox('**Visualization**'):
        if st.sidebar.checkbox('Count plot'):
            st.divider()
            sns.set(rc={"figure.figsize":(25, 10)})
            column_count = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Embarked']
            column_count_plot = st.sidebar.selectbox("Choose a column to plot count", column_count)
            hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue)", [None, 'Survived', 'Pclass', 'SibSp', 'Parch', 'Embarked'])
            count_fig = sns.countplot(x=column_count_plot,data=df,hue=hue_opt, width = 0.5)
            st.pyplot()
        if st.sidebar.checkbox('Histogram'):
            st.divider()
            sns.set(rc={"figure.figsize":(20, 10)})
            column_hist_plot = st.sidebar.selectbox("Choose a column to plot", ['Age', 'Fare'])
            hist_fig = sns.histplot(df[column_hist_plot], kde = True)
            st.pyplot()
        if st.sidebar.checkbox('Factor plot'):
            st.divider()
            column_hist_plot = st.sidebar.selectbox("Choose a column to plot", ['Sex','Pclass', 'SibSp', 'Parch', 'Embarked'])
            hue_opt = st.sidebar.selectbox("Optional categorical variables (catplot hue)", [None, 'Sex','Pclass', 'SibSp', 'Parch', 'Embarked'])
            cat_fig = sns.catplot(x = column_hist_plot, y = 'Survived', data = df, kind = 'point', hue  = hue_opt)
            col1, col2 = st.columns([1,1])
            col1.pyplot()

#==================More Visualization========================
data  = pd.read_csv('after_preprocessing.csv')
X = data.iloc[:, 1: ]
y = data.iloc[:, 0]
scaler  = MinMaxScaler()
data_rescaled = scaler.fit_transform(X)
if selected == 'More Visualization':
    if st.checkbox(":blue[Show Preprocessed Data]", False):
           st.subheader(':red[Preprocessed data]')
           st.write(data)
    if st.sidebar.checkbox('**3-D Visualization**'):
        x_dimension = st.sidebar.selectbox("Choose a column to plot at x-axis:", data.columns)
        y_dimension = st.sidebar.selectbox("Choose a column to plot at y-axis:", data.columns)
        z_dimension = st.sidebar.selectbox("Choose a column to plot at z-axis:", data.columns)
        color = st.sidebar.selectbox("Choose a column to color the species:", data.columns)
        fig = px.scatter_3d(data_frame= data,
                            x = x_dimension,
                            y= y_dimension,
                            z = z_dimension,
                            color = color)
        st.plotly_chart(fig, theme= None)
    if st.sidebar.checkbox('**Dimensionality Reduction**'):
        if st.checkbox(":blue[Show Scree Plot in PCA]", True):
            pca_nD = PCA(n_components = 6) # 6D
            principalComponents_titanic_nD  = pca_nD.fit_transform(data_rescaled)
            PC_values = np.arange(pca_nD.n_components_) + 1
            plt.plot(PC_values, pca_nD.explained_variance_ratio_, 'o-', linewidth = 2, color = 'b')
            plt.title('Scree plot')
            plt.xlabel('Principal Component')
            plt.ylabel('% Explained Variance')
            st.pyplot()
        if st.sidebar.checkbox('Principal Component Analysis in 2D'):
            pca_2d = PCA(n_components= 2)
            principalComponents_titanic = pca_2d.fit_transform(data_rescaled)
            principal_df_2d = pd.DataFrame(data = principalComponents_titanic, columns = ['Principal Component 1', 'Principal Component 2'])
            plt.figure()
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA of Titanic Dataset')
            targets = [0,1]
            colors = ['r', 'g']
            for target, color in zip(targets, colors):
                indicesToKeep = y == target
                plt.scatter(principal_df_2d.loc[indicesToKeep, 'Principal Component 1'],
                            principal_df_2d.loc[indicesToKeep, 'Principal Component 2'], 
                            c = color)
            st.pyplot()
        if st.sidebar.checkbox('Principal Component Analysis in 3D'):
            pca_3d = PCA(n_components= 3)
            principal_titanic_3D = pca_3d.fit_transform(data_rescaled)
            principal_titanic_3D_df = pd.DataFrame(data = principal_titanic_3D, columns = ['principal component 1', 'principal component 2','principal component 3'])
            principal_titanic_3D_df['labels'] = y
            fig = px.scatter_3d(principal_titanic_3D_df,
                    x  = 'principal component 1', 
                    y = 'principal component 2',
                    z = 'principal component 3', 
                    color = 'labels',
                    title = 'PCA in 3D')
            st.plotly_chart(fig, theme = None)
        if st.sidebar.checkbox('T-SNE in 2D'):
            perplexity = st.slider('Perplexity', min_value= 2, max_value = 500)
            tsne_2D  = TSNE(n_components= 2, perplexity= perplexity).fit_transform(data_rescaled)
            tsne_2D_df = pd.DataFrame(tsne_2D, columns = ['Component 1', 'Component 2'])
            tsne_2D_df['labels'] = y 
            tsne_2d_fig =  px.scatter(data_frame=  tsne_2D_df, x = 'Component 1', y = 'Component 2', color  = 'labels')
            st.plotly_chart(tsne_2d_fig, theme = None)
        if st.sidebar.checkbox('T-SNE in 3D'):
            perplexity = st.slider('Perplexity for 3D', min_value= 2, max_value = 500)
            tsne_3D  = TSNE(n_components= 3, perplexity= perplexity).fit_transform(data_rescaled)
            tsne_3D_df = pd.DataFrame(tsne_3D, columns = ['Component 1', 'Component 2', 'Component 3'])
            tsne_3D_df['labels'] = y 
            tsne_3d_fig =  px.scatter_3d(data_frame=  tsne_3D_df, x = 'Component 1', y = 'Component 2', z = 'Component 3', color  = 'labels')
            st.plotly_chart(tsne_3d_fig, theme = None) 
   #==================Prediction===============================
if selected == 'Prediction':
    st.title(':red[Predicting the Survival of Titanic Passengers]')
    # loading the trained model:
    pickle_in = open('classifier.pkl', 'rb')
    classifier = pickle.load(pickle_in)
    # Take input from user:
    Pclass = st.selectbox('Ticket class', (1,2,3))
    if Pclass == 1: st.write('This ticket is  highest class')
    elif  Pclass == 2: st.write('This ticket is 2nd class')
    else: st.write('This ticket is lowest class')

    Age  = st.number_input('Age',  min_value = 0 , max_value = 150)

    Fare = st.number_input('Fare', min_value = 0, max_value = 515)
    if Fare < 7.91: Fare_cat  = 0, st.write('This ticket is very cheap')
    elif Fare < 14.45: Fare_cat = 1, st.write('This ticket is cheap')
    elif Fare <  31: Fare_cat = 2, st.write('This ticket is reasonable')
    else: Fare_cat = 3, st.write('This ticket is expensive')

    Sex = st.selectbox('Sex', ('Female', 'Male'))

    IsFemale = 1 if Sex == 'Female' else 0

    SibSp = st.number_input('Number of Siblings / Spouses', min_value= 0, max_value = 16)
    Parent = st.number_input('Number of family relations', min_value = 0 , max_value = 12)
    Family_member = SibSp + Parent
    st.write('Family members are:', Family_member) 
    
    input_data  = pd.DataFrame([[Pclass, Age, Fare_cat[0], IsFemale, Family_member]], columns= ['Pclass', 'Age',  'Fare_cat', 'IsFemale', 'Family_member'])
    
    if st.button('Predict'):
        result = classifier.predict(input_data)
        if result == 1:
            st.success('This person is alive' ,icon = '✅')
        else: st.success('This person did not survive', icon = '🚨')
    





    




        




      


