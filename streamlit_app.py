import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import figure_factory

df = pd.read_csv("climate_change_dataset.csv")

# Hide the Streamlit header and footer
def hide_header_footer():
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_header_footer()

st.set_page_config(
    page_title="Climate Change Lab App ", layout="wide", page_icon="images/climage-change-icon.png"
)

# navigation dropdown
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])

## select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Wine Quality","Real Estate"])
## if select_dataset == "Wine Quality":
   ## df = pd.read_csv("wine_quality_red.csv")
## else: 
   ## df = pd.read_csv("real_estate.csv")

st.title("Climate Change Prediction 🔥")

# INTRODUCTION PAGE
if app_mode == "Introduction": 
    st.image("images/drid-polar-bear.jpg", use_container_width=True)

    st.markdown("### Introduction")
    st.write("Climate change represents one of the most significant risks to global economic stability and environmental sustainability. This dashboard serves as a strategic tool for stakeholders - governments, environmental agencies, and corporations - to monitor critical climate indicators, assess risks, and track the effectiveness of sustainability initiatives.")

    st.markdown("### Objectives")
    st.write("""
    - Identify long-term patterns in climate data (2000 - 2024) and visualize trends.
    - Understand the relationship between Renewable Energy adoption and CO2 reduction.
    - Pinpoint countries or regions with high vulnerability (e.g., high Sea Level Rise).
    """)

    st.markdown("### Dataset")
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    col1.markdown(" **Year** ")
    col1.markdown("The year in which the data was recorded (2000–2024).")

    col2.markdown(" **Country** ")
    col2.markdown("The country or region for which the climate data is recorded.")

    col3.markdown(" **Avg Temperature (°C)** ")
    col3.markdown("Average annual temperature in degrees Celsius for the country.")

    col4.markdown(" **CO2 Emissions (Tons/Capita)** ")
    col4.markdown("CO₂ emissions per person measured in metric tons.")

    col5.markdown(" **Sea Level Rise (mm)** ")
    col5.markdown("Annual sea-level rise in millimeters for coastal regions.")

    col6.markdown(" **Rainfall (mm)** ")
    col6.markdown("Total annual rainfall measured in millimeters.")

    col7.markdown(" **Population** ")
    col7.markdown("Population of the country in that year.")

    col8.markdown(" **Renewable Energy (%)** ")
    col8.markdown("Percent of total energy consumption from renewables (e.g., solar, wind).")

    col9.markdown(" **Extreme Weather Events** ")
    col9.markdown("Number of extreme events (floods, storms, wildfires) reported that year.")

    col10.markdown(" **Forest Area (%)** ")
    col10.markdown("Percent of the country's land area covered by forests.")
        
    st.markdown("A preview of the dataset is shown below:")    
    st.dataframe(df.head())
    st.write("Source: https://www.kaggle.com/datasets/bhadramohit/climate-change-dataset")

    st.markdown("### Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    if totalmiss <= 30:
        st.success("Looks good! We have less than 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")

    st.markdown("### Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    if completeness >= 0.80:
        st.success("Looks good! We have a completeness ratio greater than 0.85.")
           
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

# VISUALIZATION PAGE
elif app_mode == "Visualization":
    st.markdown("### Visualization")

    list_vars = df.columns
    
########### Prediction:
if app_mode == "Prediction":
    st.markdown("### Prediction")

    # Target column: temperature
    target_col = "Avg Temperature (°C)"
    st.write(f"Target variable: **{target_col}**")

    # Choose train set size (proportion of data used for training)
    train_size = st.sidebar.number_input(
        "Train Set Size (proportion)",
        min_value=0.10,
        max_value=0.90,
        step=0.05,
        value=0.70
    )

    # Use only numeric columns as possible features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Remove target from feature candidates
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    st.markdown("#### Select Explanatory Variables (features)")
    feature_cols = st.multiselect(
        "Choose features to use in the model",
        numeric_cols,
        default=numeric_cols  # by default use all numeric features except target
    )

    if not feature_cols:
        st.warning("Please select at least one feature.")
    else:
        # X = features, y = target
        X = df[feature_cols]
        y = df[target_col]

        # Show a preview of the data used
        col1, col2 = st.columns(2)
        col1.subheader("Feature Columns (top 10 rows)")
        col1.write(X.head(10))

        col2.subheader("Target Column (top 10 rows)")
        col2.write(y.head(10))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=1 - train_size,
            random_state=42
        )

        # Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions on test set
        y_pred = model.predict(X_test)

        # ==========================
        #   Metrics and results
        # ==========================
        st.markdown("### Model Performance")

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse:.2f}")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("R² Score", f"{r2:.3f}")

        st.markdown("### Actual vs Predicted Temperature (sample)")
        results_df = (
            pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            .reset_index(drop=True)
        )
        st.dataframe(results_df.head(20))

    
    






st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")

