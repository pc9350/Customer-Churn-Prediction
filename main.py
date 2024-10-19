import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

# Set the page background to black
ut.set_page_style()

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get("GROQ_API_KEY"),
)

def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)

xgboost_model = load_model("xgb_model.pkl")

naive_bayes_model = load_model("nb_model.pkl")

random_forest_model = load_model("rf_model.pkl")

decision_treen_model = load_model("dt_model.pkl")

svm_model = load_model("svm_model.pkl")

knn_model = load_model("knn_model.pkl")

voting_classifier_model = load_model("voting_clf.pkl")

xgbost_SMOTE_model = load_model("xgboost-SMOTE.pkl")

xgboost_featureEngineered_model = load_model("xgboost-featureEngineered.pkl")

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):

  input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "hasCrCard": int(has_credit_card),
    "IsActiveMember": int(is_active_member),
    "EstimatedSalary": estimated_salary,
    "Geography_France": 1 if location == "France" else 0,
    "Geography_Germany": 1 if location == "Germany" else 0,
    "Geography_Spain": 1 if location == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predicitons(input_df, input_dict):

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, ue_container_width=True)

  return avg_probability


def explain_prediction(probability, input_dict, surname):

  prompt = f"""You are an expert data scientist at a bank , where you specialize in interpreting and explaining predictions of machine learning models.
  
  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.
  
  Here is the customer's information:
  {input_dict}
  
  Here are the machine learning model's top 10 most important features for predicting churn:
  
          Feature |  Importance
      --------------------------------   
        NumOfProdcuts | 0.323888
       IsActiveMember | 0.164146
                  Age | 0.109550
    Geography_Germany | 0.091373
              Balance | 0.052786
     Geography_France | 0.046463
        Fender_Female | 0.045283
      Geography_Spain | 0.036855
          CreditScore | 0.035005
      EstimatedSalary | 0.032655
            HasCrCard | 0.031940
               Tenure | 0.030054
          Gender_Male | 0.000000


    {pd.set_option('display.max_columns', None)}

    Here are summry statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summry statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

- If the customer ahs over a 40% risk of churning, generate a 3 sentence explanation of why they are at a risk of churning.
- If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at a risk of churning.
- Your explanaton should be based on the customer's information, the summary statistics of churned and non-churned customers, adn the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything like" Based on the machine learning model's prediction and top 10 most imoportant features", or something about the customer's specific information like credit score and age andtenure or any other inormation, just explain the prediction.
        
        
  """

  print("EXPLANATION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )
  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""
    You are a manager at HS Bank, responsible for customer retention. A customer named {surname} has shown a {round(probability * 100, 1)}% chance of churning.

    Customer Information:
    {input_dict}

    Churn Risk Explanation:
    {explanation}

    Based on this, draft a personalized email encouraging the customer to remain with the bank. Highlight tailored incentives to improve their loyalty, such as:
    - Exclusive offers on products
    - Personalized financial advice
    - Reduced fees or special interest rates
    - Loyalty rewards or cashback programs

    Ensure the tone is friendly and customer-centric, focusing on building long-term value. Avoid mentioning the churn probability or any machine learning predictions.
    Also do not mention anything after the email signature. End the email with an email sign off like warm regards or kind regards or something else and then after that in the next line "HS Bank's Customer Service Team".
    The email should end after these instructions. There shouldn't ne any text after this.
    """
  raw_response = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[{
        "role": "user",
        "content": prompt
      }],
    )
  
  print("\n\nEMAIL PROMPT", prompt)
  
  return raw_response.choices[0].message.content 

def calculate_percentiles(selected_customer, df):
    percentiles = {}
    metrics = ['NumOfProducts', 'Balance', 'EstimatedSalary', 'Tenure', 'CreditScore']
    
    for metric in metrics:
        # Calculate the percentile of the selected customer in comparison to the whole dataset
        percentiles[metric] = (df[metric] < selected_customer[metric]).mean() * 100
    
    return percentiles


st.title("Customer Churn Prediciton")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected Customer ID", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]
  print("Selected Surname", selected_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:
    credit_Score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore']))

    location = st.selectbox(
        "Location", ["Spain", "France", "Germany"],
        index=["Spain", "France", "Germany"].index(
          selected_customer['Geography']))

    gender = st.radio("Gender", ["Male", "Female"],
                     index=0 if selected_customer['Gender'] == 'Male' else 1)

    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value=int(selected_customer['Age']))

    tenure = st.number_input(
      "Tenure (years)",
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure']))

  with col2:
    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer['Balance']))

    num_products = st.number_input(
      "Number of Products",
      min_value=1,
      max_value=10,
      value=int(selected_customer['NumOfProducts']))
  
    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer['HasCrCard']))

    is_active_member = st.checkbox(
      "Is Active Member",
      value=bool(selected_customer['IsActiveMember']))
  
    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary']))
    

  # Calculate percentiles for selected customer
  percentiles = calculate_percentiles(selected_customer, df)
    
  input_df, input_dict = prepare_input(credit_Score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

  avg_probability = make_predicitons(input_df, input_dict)

  # Create and display the percentile chart
  st.subheader("Customer Percentiles")
  fig_percentiles = ut.create_percentile_chart(percentiles)
  st.plotly_chart(fig_percentiles, use_container_width=True)

  explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

  st.markdown("---")

  st.subheader("Explanation of Prediction")

  st.markdown(explanation)

  email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

  st.markdown("---")

  st.subheader("Personalized Email")

  st.markdown(email)
