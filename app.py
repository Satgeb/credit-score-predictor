import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00cc00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    """Load the trained Random Forest pipeline model"""
    try:
        model = joblib.load('rf_customer.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Header
st.markdown('<h1 class="main-header">💳 Credit Score Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📋 Customer Information")

# Helper function for safe number input
def safe_number_input(label, min_val, max_val, default_val, step=1.0, help_text=""):
    return st.sidebar.number_input(
        label, 
        min_value=float(min_val), 
        max_value=float(max_val), 
        value=float(default_val),
        step=step,
        help=help_text
    )

# Collect all inputs matching your dataset columns
with st.sidebar.form("customer_data_form"):
    st.subheader("Personal Details")
    
    month = st.selectbox(
        "Month", 
        ['January', 'February', 'March', 'April', 'May', 'June', 
         'July', 'August', 'September', 'October', 'November', 'December']
    )
    
    age = safe_number_input("Age", 18, 100, 30, 1.0, "Customer age in years")
    
    ssn = st.text_input("SSN (Last 4 digits)", "1234", help="Last 4 digits of SSN")
    
    occupation = st.selectbox(
        "Occupation",
        ['Scientist', 'Engineer', 'Accountant', 'Doctor', 'Manager', 
         'Teacher', 'Developer', 'Lawyer', 'Entrepreneur', 'Other']
    )
    
    annual_income = safe_number_input("Annual Income ($)", 10000, 500000, 50000, 1000.0)
    monthly_salary = safe_number_input("Monthly In-hand Salary ($)", 1000, 50000, 4000, 100.0)
    
    st.subheader("Banking Information")
    
    num_bank_accounts = st.slider("Number of Bank Accounts", 1, 15, 3)
    num_credit_cards = st.slider("Number of Credit Cards", 0, 15, 2)
    interest_rate = safe_number_input("Interest Rate (%)", 0, 50, 15, 0.5)
    num_loans = safe_number_input("Number of Loans", 0, 10, 1, 1.0)
    
    type_of_loan = st.multiselect(
        "Type of Loan(s)",
        ['Personal Loan', 'Auto Loan', 'Home Loan', 'Student Loan', 
         'Credit-Builder Loan', 'Debt Consolidation', 'Payday Loan'],
        default=['Personal Loan']
    )
    
    st.subheader("Payment Behavior")
    
    delay_from_due = st.slider("Delay from Due Date (days)", 0, 100, 0)
    num_delayed_payments = safe_number_input("Number of Delayed Payments", 0, 100, 0, 1.0)
    
    changed_credit_limit = safe_number_input("Changed Credit Limit", -10000, 50000, 0, 100.0)
    num_credit_inquiries = safe_number_input("Number of Credit Inquiries", 0, 50, 2, 1.0)
    
    outstanding_debt = safe_number_input("Outstanding Debt ($)", 0, 200000, 5000, 100.0)
    credit_utilization = safe_number_input("Credit Utilization Ratio", 0.0, 100.0, 30.0, 0.1)
    
    credit_history_age = st.text_input("Credit History Age", "15 Years and 3 Months")
    
    payment_min_amount = st.selectbox("Payment of Min Amount", ['Yes', 'No', 'NM'])
    
    total_emi = safe_number_input("Total EMI per Month ($)", 0, 10000, 500, 50.0)
    amount_invested = safe_number_input("Amount Invested Monthly ($)", 0, 20000, 1000, 100.0)
    
    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
         'Low_spent_Small_value_payments', 'High_spent_Large_value_payments']
    )
    
    monthly_balance = safe_number_input("Monthly Balance ($)", -5000, 50000, 3000, 100.0)
    
    submitted = st.form_submit_button("🔮 Predict Credit Score")

# Main content area
if submitted and model is not None:
    # Prepare input data as DataFrame (matching training format)
    input_data = pd.DataFrame({
        'Month': [month],
        'Age': [age],
        'SSN': [ssn],
        'Occupation': [occupation],
        'Annual_Income': [annual_income],
        'Monthly_Inhand_Salary': [monthly_salary],
        'Num_Bank_Accounts': [num_bank_accounts],
        'Num_Credit_Card': [num_credit_cards],
        'Interest_Rate': [interest_rate],
        'Num_of_Loan': [num_loans],
        'Type_of_Loan': [', '.join(type_of_loan) if type_of_loan else 'No Loan'],
        'Delay_from_due_date': [delay_from_due],
        'Num_of_Delayed_Payment': [num_delayed_payments],
        'Changed_Credit_Limit': [changed_credit_limit],
        'Num_Credit_Inquiries': [num_credit_inquiries],
        'Outstanding_Debt': [outstanding_debt],
        'Credit_Utilization_Ratio': [credit_utilization],
        'Credit_History_Age': [credit_history_age],
        'Payment_of_Min_Amount': [payment_min_amount],
        'Total_EMI_per_month': [total_emi],
        'Amount_invested_monthly': [amount_invested],
        'Payment_Behaviour': [payment_behaviour],
        'Monthly_Balance': [monthly_balance]
    })
    
    # Display input summary
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Input Summary")
        st.dataframe(input_data.T, use_container_width=True)
    
    with col2:
        try:
            # Make prediction using the pipeline
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
            
            # Map prediction to credit score category
            # Assuming your model predicts: 0=Good, 1=Standard, 2=Poor (adjust as per your encoding)
            score_mapping = {0: "Good", 1: "Standard", 2: "Poor"}
            risk_colors = {"Good": "risk-low", "Standard": "risk-medium", "Poor": "risk-high"}
            
            predicted_score = score_mapping.get(prediction[0], prediction[0])
            color_class = risk_colors.get(predicted_score, "risk-medium")
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Predicted Credit Score Category</h3>
                    <h2 class="{color_class}">{predicted_score}</h2>
                    <p>Prediction made on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution if available
            if prediction_proba is not None:
                st.subheader("📈 Prediction Probabilities")
                proba_df = pd.DataFrame({
                    'Credit Score': ['Good', 'Standard', 'Poor'],
                    'Probability': prediction_proba[0]
                })
                
                # Create probability bar chart
                import plotly.express as px
                fig = px.bar(
                    proba_df, 
                    x='Credit Score', 
                    y='Probability',
                    color='Credit Score',
                    color_discrete_map={'Good': '#00cc00', 'Standard': '#ffa500', 'Poor': '#ff4b4b'},
                    text=proba_df['Probability'].apply(lambda x: f'{x:.2%}')
                )
                fig.update_layout(yaxis_range=[0, 1], showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                max_proba = np.max(prediction_proba[0])
                st.info(f"**Confidence Level:** {max_proba:.2%}")
                
                # Recommendations based on prediction
                st.subheader("💡 Recommendations")
                if predicted_score == "Poor":
                    st.error("""
                    - **Immediate Action Required:** High risk of default detected
                    - Consider debt consolidation
                    - Set up automatic payments to avoid delays
                    - Reduce credit utilization below 30%
                    """)
                elif predicted_score == "Standard":
                    st.warning("""
                    - **Moderate Risk:** Room for improvement
                    - Maintain consistent payment history
                    - Avoid new credit inquiries
                    - Increase monthly investments if possible
                    """)
                else:
                    st.success("""
                    - **Excellent Standing:** Keep up the good work!
                    - Continue current financial habits
                    - Consider premium credit products
                    - Maintain low credit utilization
                    """)
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check that all input values match the expected format from training data.")

else:
    # Initial state
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Credit Score** to get started.")
    
    # Sample data display
    st.subheader("📋 Expected Input Format")
    sample_data = pd.DataFrame({
        'Month': ['January'],
        'Age': [30.0],
        'SSN': ['1234'],
        'Occupation': ['Engineer'],
        'Annual_Income': [50000.0],
        'Monthly_Inhand_Salary': [4000.0],
        'Num_Bank_Accounts': [3],
        'Num_Credit_Card': [2],
        'Interest_Rate': [15],
        'Num_of_Loan': [1.0],
        'Type_of_Loan': ['Personal Loan'],
        'Delay_from_due_date': [0],
        'Num_of_Delayed_Payment': [0.0],
        'Changed_Credit_Limit': [0.0],
        'Num_Credit_Inquiries': [2.0],
        'Outstanding_Debt': [5000.0],
        'Credit_Utilization_Ratio': [30.0],
        'Credit_History_Age': ['15 Years and 3 Months'],
        'Payment_of_Min_Amount': ['Yes'],
        'Total_EMI_per_month': [500.0],
        'Amount_invested_monthly': [1000.0],
        'Payment_Behaviour': ['High_spent_Small_value_payments'],
        'Monthly_Balance': [3000.0]
    })
    st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Credit Score Prediction System | Powered by Random Forest Classifier*")