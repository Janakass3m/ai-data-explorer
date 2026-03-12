import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.title("AI Data Explorer")
st.caption("Upload a CSV to explore summary statistics, visualizations, and AI-generated insights.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Preview")
    st.dataframe(df.head())

    st.write("### Shape")
    st.write(df.shape)

    st.write("### Columns")
    st.dataframe(pd.DataFrame({"Column Name": df.columns}))

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Data Types")
    st.write(df.dtypes)

    st.write("### Numeric Summary")
    st.write(df.describe())

    st.write("## Interactive Visualizations")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) > 0:
        st.write("### Histogram")
        hist_col = st.selectbox(
            "Choose a numeric column for histogram",
            numeric_cols,
            key="hist_col"
        )
        fig_hist = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
        st.plotly_chart(fig_hist, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.write("### Scatterplot")
        x_col = st.selectbox(
            "Choose x-axis",
            numeric_cols,
            index=0,
            key="x_axis"
        )
        y_default_index = 1 if len(numeric_cols) > 1 else 0
        y_col = st.selectbox(
            "Choose y-axis",
            numeric_cols,
            index=y_default_index,
            key="y_axis"
        )

        if x_col == y_col:
            st.warning("Please choose two different variables for the scatterplot.")
        else:
            scatter_df = df[[x_col, y_col]].dropna()
            fig_scatter = px.scatter(
                scatter_df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                render_mode="svg"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    if len(categorical_cols) > 0:
        st.write("### Bar Chart")
        cat_col = st.selectbox(
            "Choose a categorical column",
            categorical_cols,
            key="cat_col"
        )
        cat_counts = df[cat_col].value_counts().reset_index()
        cat_counts.columns = [cat_col, "Count"]
        fig_bar = px.bar(
            cat_counts,
            x=cat_col,
            y="Count",
            title=f"Counts of {cat_col}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.write("## Correlation Analysis")

    numeric_df = df.select_dtypes(include=["number"])

    if len(numeric_df.columns) >= 2:
        target_col = st.selectbox(
            "Choose a numeric variable to analyze correlations",
            numeric_df.columns,
            key="corr_target"
        )

        corr_series = numeric_df.corr(numeric_only=True)[target_col].drop(labels=[target_col]).dropna()
        corr_df = corr_series.reindex(
            corr_series.abs().sort_values(ascending=False).index
        ).reset_index()
        corr_df.columns = ["Variable", "Correlation"]

        st.write(f"### Variables most correlated with {target_col}")
        st.dataframe(corr_df, use_container_width=True)

        if not corr_df.empty:
            top_var = corr_df.iloc[0]["Variable"]
            top_corr = corr_df.iloc[0]["Correlation"]

            st.success(
                f"The variable most correlated with {target_col} is {top_var} "
                f"(correlation = {top_corr:.3f})."
            )

            st.write("### AI Explanation of Correlation Result")

            if "corr_explanation" not in st.session_state:
                st.session_state.corr_explanation = None

            if st.button("Explain Top Correlation", key="explain_corr"):
                api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

                if not api_key:
                    st.error("OpenAI API key not found.")
                else:
                    try:
                        client = OpenAI(api_key=api_key)

                        prompt = f"""
You are a data analysis assistant.

A dataset was analyzed for correlations.

Target variable: {target_col}
Most correlated variable: {top_var}
Correlation value: {top_corr:.3f}

Explain in plain English:
1. What this correlation means
2. Whether it seems weak, moderate, or strong
3. Why correlation does not necessarily imply causation
4. One useful next step for further analysis

Keep the explanation concise and practical.
"""

                        with st.spinner("Generating explanation..."):
                            response = client.responses.create(
                                model="gpt-4o-mini",
                                input=prompt
                            )

                        st.session_state.corr_explanation = response.output_text

                    except Exception as e:
                        st.error(f"Could not generate explanation: {e}")

            if st.session_state.corr_explanation:
                st.markdown(st.session_state.corr_explanation)

    st.write("## AI Insights")

    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = None

    if st.button("Generate AI Insights"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error("OpenAI API key not found. Please add it to your .env file.")
        else:
            try:
                client = OpenAI(api_key=api_key)

                numeric_summary = df.describe(include="all").fillna("").to_string()

                summary_text = f"""
Dataset shape: {df.shape}

Columns:
{df.columns.tolist()}

Data types:
{df.dtypes.to_string()}

Missing values:
{df.isnull().sum().to_string()}

Summary statistics:
{numeric_summary}
"""

                prompt = f"""
You are an AI data analysis assistant.

Based on the dataset summary below, provide:

1. A short plain-English description of what this dataset appears to contain
2. Three interesting questions someone could explore
3. Three patterns or insights worth investigating
4. Two modeling ideas that could be useful

Keep the response practical, clear, and organized with section headers.
Do not make up facts that are not supported by the summary.

Dataset summary:
{summary_text}
"""

                with st.spinner("Generating AI insights..."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                st.session_state.ai_insights = response.output_text

            except Exception as e:
                st.error(f"Could not generate AI insights: {e}")

    if st.session_state.ai_insights:
        st.markdown(st.session_state.ai_insights)

    st.write("## Ask Questions About Your Dataset")

    if "ai_answer" not in st.session_state:
        st.session_state.ai_answer = None

    user_question = st.text_input("Ask a question about this dataset")

    if user_question:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error("OpenAI API key not found.")
        else:
            try:
                client = OpenAI(api_key=api_key)

                dataset_summary = f"""
Dataset shape: {df.shape}

Columns:
{df.columns.tolist()}

Data types:
{df.dtypes.to_string()}

Missing values:
{df.isnull().sum().to_string()}

Summary statistics:
{df.describe(include='all').fillna('').to_string()}
"""

                prompt = f"""
You are an AI data analyst helping a user understand their dataset.

Dataset summary:
{dataset_summary}

User question:
{user_question}

Answer the question using only the dataset information available.
If the question cannot be answered directly, suggest how the user could analyze the data.
"""

                with st.spinner("Analyzing dataset..."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                st.session_state.ai_answer = response.output_text

            except Exception as e:
                st.error(f"Error generating response: {e}")

    if st.session_state.ai_answer:
        st.markdown(st.session_state.ai_answer)