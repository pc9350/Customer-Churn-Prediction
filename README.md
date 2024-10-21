# Customer Churn Prediction App

A full-stack application built using Streamlit that predicts customer churn using machine learning models and automates personalized communication for retention strategies. This project is designed to help businesses identify customers at risk of churning and take proactive measures to retain them.

### [Live Demo](https://churncrunch.streamlit.app/)

## Features

- **Churn Prediction**: Uses various machine learning models like XGBoost, Random Forest, and K-Nearest Neighbors to predict the likelihood of customer churn.
- **Automated Emails**: Integrates the **Groq API** to generate personalized emails and explanations based on churn prediction results.
- **Data Visualization**: Displays churn probabilities and key customer metrics using interactive visualizations created with **Plotly**.
- **Enhanced Model Accuracy**: Improved the model's accuracy from **75% to 85%** using **feature engineering** and **SMOTE** to handle imbalanced data.

## Tech Stack

- **Frontend**: Streamlit, Replit
- **Backend**: Machine learning models trained on a Kaggle dataset
- **APIs**: Groq API for automated email generation and prediction explanations
- **Visualization**: Plotly for customer metric visualizations
- **Machine Learning**: XGBoost, Random Forest, K-Nearest Neighbors, SMOTE, Feature Engineering

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:
- **Python 3.x**
- **Streamlit**
- **Pandas**
- **Scikit-learn**
- **Plotly**
- **Pickle**
- **OpenAI** and **Groq API**

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/pc9350/Customer-Churn-Prediction.git
    cd customer-churn-prediction-app
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root and add your **Groq API** and **OpenAI API** keys:
    ```
    GROQ_API_KEY=your_groq_api_key
    OPENAI_API_KEY=your_openai_key
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

### Dataset

This project uses a customer churn dataset from Kaggle. You can download the dataset [here](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data) and place it in the project directory. Make sure the file is named `churn.csv`.

## Usage

1. **Select a customer** from the dataset using the dropdown menu.
2. **Input or modify customer details** (credit score, balance, tenure, etc.) to generate a prediction.
3. **View predictions**: The app will display churn probability along with key customer metrics visualized through graphs and charts.
4. **Automated emails**: Based on the prediction, an email template will be generated using the **Groq API** to encourage customer retention.

## Model Training

The models used for predictions (XGBoost, Random Forest, KNN) were trained on the Kaggle dataset. The training process involved:
- **Feature Engineering**: Enhancing the dataset by adding relevant features.
- **SMOTE**: Applied to handle data imbalance and improve the performance of the models.
- Models were saved as `.pkl` files and loaded during runtime for predictions.

## Visualizations

The app provides several visualizations:
- **Gauge Chart**: Displays the overall churn probability of the selected customer.
- **Percentile Graph**: Shows how the customer compares to others across metrics like balance, credit score, and tenure.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request if you'd like to improve or expand the project.

## Contact

For any inquiries or feedback, feel free to reach out via:
- **Email**: pranavchhabra88@gmail.com
- **LinkedIn**: [Pranav Chhabra](https://linkedin.com/in/pranavchhabra)

