import streamlit as st
import pandas as pd

# Title
st.title("Basic Streamlit App")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.subheader("Data Preview:")
    st.write(df)

    # Basic statistics
    st.subheader("Basic Statistics:")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Column names:", df.columns.tolist())
    st.write("Summary statistics:", df.describe())

    st.subheader("Summary Statistics for Each Column:")

    for column in df.columns:
        st.write(f"**{column}**:")
        st.write(f"Count: {df[column].count()}")
        st.write(f"Mean: {df[column].mean()}")
        st.write(f"Standard Deviation: {df[column].std()}")
        st.write(f"Minimum: {df[column].min()}")
        st.write(f"Maximum: {df[column].max()}")
        percentiles = df[column].describe(percentiles=[0.25, 0.5, 0.75])
        st.write(f"25th Percentile: {percentiles['25%']}")
        st.write(f"Median (50th Percentile): {percentiles['50%']}")
        st.write(f"75th Percentile: {percentiles['75%']}")
        st.write("\n")

    # Select specific columns for display
    selected_columns = st.multiselect("Select columns to display:", df.columns.tolist())
    if selected_columns:
        st.write("Selected Columns:")
        st.write(df[selected_columns])
