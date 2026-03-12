# AI Data Explorer

An interactive web app that allows users to upload a dataset and instantly explore it using statistics, visualizations, and AI-powered insights.

Live Demo  
https://ai-data-explorer.streamlit.app/

GitHub Repository  
https://github.com/Janakass3m/ai-data-explorer


## Overview

AI Data Explorer is a lightweight tool designed to make exploratory data analysis easier and faster.

Users can upload any CSV dataset and immediately:

• view summary statistics  
• generate interactive visualizations  
• analyze correlations between variables  
• receive AI-generated insights about the dataset  
• ask natural language questions about the data  

The app combines traditional data analysis with AI assistance to help users quickly understand unfamiliar datasets.


## Features

### Dataset Overview
Automatically displays:

- dataset shape
- column names
- missing values
- data types
- statistical summaries

### Interactive Visualizations

Users can generate multiple visualizations instantly:

- Histogram for numeric distributions
- Scatterplot between numeric variables
- Bar charts for categorical variables

Built with Plotly for interactive exploration.

### Correlation Analysis

The app automatically computes correlations between numeric variables and identifies which features are most strongly related.

Users can select a target variable and instantly see which variables have the highest correlation.

### AI-Powered Insights

Using the OpenAI API, the app generates:

- a plain-English dataset description
- interesting questions to explore
- potential insights or patterns
- ideas for modeling approaches

### Natural Language Data Questions

Users can ask questions about the dataset such as:

- "What are the most important variables in this dataset?"
- "Are there any potential biases in this dataset?"
- "Which variables might cause data quality issues?"
- "What interesting research questions could this dataset answer?"

The AI provides explanations or suggests analyses based on the dataset summary.


## Tech Stack

Python  
Streamlit  
Pandas  
Plotly  
OpenAI API  
python-dotenv


## Example Use Cases

- quick exploratory data analysis
- teaching data science concepts
- exploring new datasets
- generating initial research questions
- identifying potential predictive features


## Running Locally

Clone the repository
