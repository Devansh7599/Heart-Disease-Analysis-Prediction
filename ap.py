import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- New Imports for AI Chatbot ---
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Set page config
st.set_page_config(
    page_title="Heart Disease Analysis & Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Make sure your CSV file is in the same directory or provide the correct path
    try:
        df = pd.read_csv('heart_disease_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'heart_disease_dataset.csv' not found.")
        st.info("Please make sure the dataset file is in the same directory as your app.py file.")
        return pd.DataFrame() # Return empty dataframe to avoid further errors

# Load model
@st.cache_resource
def load_model():
    # Make sure your model file is in the same directory or provide the correct path
    try:
        model = joblib.load('best_heart_disease_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'best_heart_disease_model.joblib' not found.")
        st.info("Please make sure the model file is in the same directory as your app.py file.")
        return None # Return None to avoid further errors

# Data preprocessing function for prediction
def preprocess_data(df):
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values - fill missing Alcohol Intake with 'None'
    df_processed['Alcohol Intake'] = df_processed['Alcohol Intake'].fillna('None')
    
    # Encode binary categorical features
    binary_cols = ['Gender', 'Smoking', 'Diabetes', 'Obesity', 'Family History', 'Exercise Induced Angina']
    for col in binary_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # One-hot encode multi-class categorical features with drop_first=True to match training
    df_processed = pd.get_dummies(df_processed, columns=['Alcohol Intake', 'Chest Pain Type'], drop_first=True)
    
    # Remove outliers using IQR for numerical features
    def remove_outliers(df, cols):
        for col in cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df
    
    num_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level', 'Blood Sugar']
    df_processed = remove_outliers(df_processed, num_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    
    return df_processed, scaler

# Function to preprocess single prediction input to match training features
def preprocess_prediction_input(input_data):
    """Preprocess input data to match exactly what the model was trained on"""
    
    # Create a copy
    df = input_data.copy()
    
    # Handle missing values
    df['Alcohol Intake'] = df['Alcohol Intake'].fillna('None')
    
    # Encode binary categorical features exactly as in training
    # NOTE: Your original code had an issue here. 'Smoking' is not binary.
    # I've updated 'Smoking' to match common ordinal encoding.
    binary_encodings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Diabetes': {'No': 0, 'Yes': 1},
        'Obesity': {'No': 0, 'Yes': 1},
        'Family History': {'No': 0, 'Yes': 1},
        'Exercise Induced Angina': {'No': 0, 'Yes': 1}
    }
    
    for col, encoding in binary_encodings.items():
        if col in df.columns:
            df[col] = df[col].map(encoding)

    # Handle 'Smoking' separately as it's ordinal
    smoking_encoding = {'Never': 0, 'Former': 1, 'Current': 2}
    if 'Smoking' in df.columns:
        df['Smoking'] = df['Smoking'].map(smoking_encoding)
    
    # One-hot encode categorical features with drop_first=True
    df = pd.get_dummies(df, columns=['Alcohol Intake', 'Chest Pain Type'], drop_first=True)
    
    # Define the exact feature list the model was trained on
    # You MUST adjust this list to match your *actual* trained model
    expected_features = [
        'Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 
        'Smoking', 'Exercise Hours', 'Family History', 'Diabetes', 'Obesity', 
        'Stress Level', 'Blood Sugar', 'Exercise Induced Angina',
        'Alcohol Intake_Moderate', 'Alcohol Intake_None', # Assuming 'Heavy' was dropped
        'Chest Pain Type_Atypical Angina', 'Chest Pain Type_Non-anginal Pain', 
        'Chest Pain Type_Typical Angina' # Assuming 'Asymptomatic' was dropped
    ]
    
    # Add missing columns with 0 values
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the expected features in the correct order
    # Ensure all features are present before reordering
    missing_from_expected = [f for f in expected_features if f not in df.columns]
    if missing_from_expected:
        st.warning(f"Warning: Model features missing from input: {missing_from_expected}")
        # Add them with 0
        for f in missing_from_expected:
            df[f] = 0
            
    df = df[expected_features]
    
    # Scale numerical features using the same scaling as training
    numerical_features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level', 'Blood Sugar', 'Exercise Hours']
    
    # IMPORTANT: These values MUST match your training data. 
    # These are illustrative. Replace with your actual values.
    # You can get these from your saved scaler or by printing df.describe() before scaling in your notebook.
    scaling_params = {
        'Age': {'mean': 52.29, 'std': 15.73},
        'Cholesterol': {'mean': 249.94, 'std': 57.91},
        'Blood Pressure': {'mean': 135.28, 'std': 26.39},
        'Heart Rate': {'mean': 79.20, 'std': 11.49},
        'Stress Level': {'mean': 5.65, 'std': 2.83},
        'Blood Sugar': {'mean': 134.94, 'std': 36.70},
        'Exercise Hours': {'mean': 5.0, 'std': 3.0} # Added this, adjust values
    }
    
    for feature in numerical_features:
        if feature in df.columns:
            # Check if params exist
            if feature in scaling_params:
                mean_val = scaling_params[feature]['mean']
                std_val = scaling_params[feature]['std']
                # Avoid division by zero if std is 0
                if std_val > 0:
                    df[feature] = (df[feature] - mean_val) / std_val
                else:
                    df[feature] = 0 # Or just (df[feature] - mean_val)
            else:
                st.warning(f"Missing scaling parameters for feature: {feature}")
    
    return df

# Main app
def main():
    st.title("❤️ Heart Disease Analysis & Prediction Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    model = load_model()

    # Check if data or model failed to load
    if df.empty or model is None:
        st.error("Application cannot start. Please check file paths for dataset and model.")
        return # Stop execution
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a section:", 
                           ["📊 Data Overview & Summary", 
                            "📈 Data Visualizations", 
                            "🔮 Heart Disease Prediction",
                            "🤖 AI Health Assistant"]) # <-- ADDED
    
    if page == "📊 Data Overview & Summary":
        show_data_overview(df)
    elif page == "📈 Data Visualizations":
        show_visualizations(df)
    elif page == "🔮 Heart Disease Prediction":
        show_prediction(df, model)
    elif page == "🤖 AI Health Assistant": # <-- ADDED
        show_chatbot_page(df) # Pass the dataframe

def show_data_overview(df):
    st.header("📊 Data Overview & Summary")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        st.metric("Total Features", len(df.columns))
    
    with col3:
        heart_disease_count = df['Heart Disease'].sum()
        st.metric("Heart Disease Cases", heart_disease_count)
    
    with col4:
        heart_disease_rate = (heart_disease_count / len(df)) * 100
        st.metric("Heart Disease Rate", f"{heart_disease_rate:.1f}%")
    
    st.markdown("---")
    
    # Dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        st.dataframe(missing_data[missing_data > 0])
    
    with col2:
        st.write("**Data Types:**")
        st.dataframe(df.dtypes.astype(str))
    
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Categorical variables summary
    st.subheader("Categorical Variables Summary")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"**{col}:**")
            st.write(df[col].value_counts())
            st.write("---")
    else:
        st.write("No categorical (object) columns found.")

def show_visualizations(df):
    st.header("📈 Data Visualizations")
    
    # Create tabs for different chart types
    tab_titles = [
        "📊 Distributions", 
        "📈 Correlation", 
        "📉 Box Plots", 
        "🔗 Pair Plot", 
        "📊 Counts", 
        "📈 Scatters", 
        "🎯 Target Analysis"
    ]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        st.subheader("Distribution of Key Features")
        
        # Select features for distribution
        features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate']
        
        fig = go.Figure()
        for i, feature in enumerate(features):
            fig.add_trace(go.Histogram(
                x=df[feature],
                name=feature,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Distribution of Key Features",
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Correlation Heatmap")
        
        # Calculate correlation for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap of Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numerical columns found for correlation.")
    
    with tabs[2]:
        st.subheader("Box Plots of Key Features")
        
        features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate']
        
        fig = go.Figure()
        for feature in features:
            fig.add_trace(go.Box(
                y=df[feature],
                name=feature,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Box Plots of Key Features",
            yaxis_title="Value",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Pair Plot Analysis")
        st.info("This plot might take a moment to load.")
        
        # Select subset of features for pair plot
        selected_features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Heart Disease']
        
        try:
            # Using seaborn for pairplot as it's often more robust
            fig = sns.pairplot(df[selected_features], hue='Heart Disease', palette={0: 'blue', 1: 'red'})
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating Pair Plot: {e}")
            st.write("Trying with Plotly...")
            try:
                fig_plotly = px.scatter_matrix(
                    df[selected_features],
                    dimensions=selected_features[:-1],
                    color='Heart Disease',
                    title="Pair Plot Analysis of Key Features",
                    color_discrete_map={0: 'blue', 1: 'red'}
                )
                fig_plotly.update_layout(height=800)
                st.plotly_chart(fig_plotly, use_container_width=True)
            except Exception as e_plotly:
                st.error(f"Plotly Pair Plot also failed: {e_plotly}")
    
    with tabs[4]:
        st.subheader("Count Plots for Categorical Variables")
        
        categorical_cols = ['Gender', 'Smoking', 'Diabetes', 'Obesity', 'Chest Pain Type', 'Family History']
        
        for col in categorical_cols:
            if col in df.columns:
                fig = px.histogram(
                    df, 
                    x=col, 
                    color='Heart Disease',
                    barmode='group',
                    title=f"Distribution of {col} by Heart Disease Status",
                    color_discrete_map={0: 'blue', 1: 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        st.subheader("Scatter Plots")
        
        # Age vs Cholesterol colored by Heart Disease
        fig1 = px.scatter(
            df, 
            x='Age', 
            y='Cholesterol', 
            color='Heart Disease',
            title="Age vs Cholesterol by Heart Disease Status",
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Blood Pressure vs Heart Rate colored by Heart Disease
        fig2 = px.scatter(
            df, 
            x='Blood Pressure', 
            y='Heart Rate', 
            color='Heart Disease',
            title="Blood Pressure vs Heart Rate by Heart Disease Status",
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tabs[6]:
        st.subheader("Target Variable Analysis")
        
        # Heart Disease distribution
        if 'Heart Disease' in df.columns:
            target_counts = df['Heart Disease'].value_counts().reset_index()
            target_counts.columns = ['Heart Disease', 'Count']
            
            fig1 = px.pie(
                target_counts, 
                names='Heart Disease', 
                values='Count', 
                title="Distribution of Heart Disease Cases",
                color='Heart Disease',
                color_discrete_map={0: 'lightblue', 1: 'red'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Heart Disease by Age groups
        if 'Age' in df.columns:
            df_copy = df.copy() # Avoid modifying cached df
            df_copy['Age_Group'] = pd.cut(df_copy['Age'], bins=[0, 40, 60, 80, 100], labels=['0-40 (Young)', '41-60 (Middle)', '61-80 (Senior)', '81+ (Elderly)'], right=False)
            
            fig2 = px.histogram(
                df_copy, 
                x='Age_Group', 
                color='Heart Disease',
                barmode='group',
                title="Heart Disease Distribution by Age Groups",
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig2, use_container_width=True)

def show_prediction(df, model):
    st.header("🔮 Heart Disease Prediction")
    
    # Show example values for low risk
    with st.expander("Click to see example patient profiles"):
        st.subheader("📋 Example Values for Low Risk (Normal)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Low Risk Example Values:**")
            st.write("""
            - **Age**: 35 years
            - **Gender**: Female
            - **Cholesterol**: 160 mg/dL
            - **Blood Pressure**: 100 mmHg
            - **Heart Rate**: 65 bpm
            - **Smoking**: Never
            - **Alcohol Intake**: None
            - **Exercise Hours**: 8 hours/week
            """)
        with col2:
            st.write("**Low Risk Example Values (continued):**")
            st.write("""
            - **Family History**: No
            - **Diabetes**: No
            - **Obesity**: No
            - **Stress Level**: 2 (Low)
            - **Blood Sugar**: 80 mg/dL
            - **Exercise Induced Angina**: No
            - **Chest Pain Type**: Asymptomatic
            """)
        
        st.markdown("---")
        
        # Show high risk example for comparison
        st.subheader("⚠️ Example Values for High Risk (for comparison)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**High Risk Example Values:**")
            st.write("""
            - **Age**: 65+ years
            - **Gender**: Male
            - **Cholesterol**: 250+ mg/dL
            - **Blood Pressure**: 140+ mmHg
            - **Heart Rate**: 90+ bpm
            - **Smoking**: Current
            - **Alcohol Intake**: Heavy
            - **Exercise Hours**: 0-2 hours/week
            """)
        with col2:
            st.write("**High Risk Example Values (continued):**")
            st.write("""
            - **Family History**: Yes
            - **Diabetes**: Yes
            - **Obesity**: Yes
            - **Stress Level**: 7-10 (High)
            - **Blood Sugar**: 150+ mg/dL
            - **Exercise Induced Angina**: Yes
            - **Chest Pain Type**: Typical Angina
            """)
    
    st.markdown("---")
    
    # Add button to fill example values
    if st.button("🔧 Fill Low Risk Example Values", help="Click to automatically fill the form with low-risk example values"):
        st.session_state.age = 35
        st.session_state.gender = "Female"
        st.session_state.cholesterol = 160
        st.session_state.blood_pressure = 100
        st.session_state.heart_rate = 65
        st.session_state.smoking = "Never"
        st.session_state.alcohol_intake = "None"
        st.session_state.exercise_hours = 8
        st.session_state.family_history = "No"
        st.session_state.diabetes = "No"
        st.session_state.obesity = "No"
        st.session_state.stress_level = 2
        st.session_state.blood_sugar = 80
        st.session_state.exercise_angina = "No"
        st.session_state.chest_pain_type = "Asymptomatic"
        st.success("✅ Low-risk example values filled!")
    
    st.markdown("Enter patient information to predict the likelihood of heart disease:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.slider("Age", min_value=18, max_value=100, value=st.session_state.get('age', 50), key='age')
            gender = st.selectbox("Gender", ["Male", "Female"], index=1 if st.session_state.get('gender', 'Female') == 'Female' else 0, key='gender')
            cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=400, value=st.session_state.get('cholesterol', 200), key='cholesterol')
            blood_pressure = st.slider("Blood Pressure (Systolic)", min_value=80, max_value=200, value=st.session_state.get('blood_pressure', 120), key='blood_pressure')
            heart_rate = st.slider("Heart Rate (bpm)", min_value=50, max_value=120, value=st.session_state.get('heart_rate', 70), key='heart_rate')
            blood_sugar = st.slider("Blood Sugar (mg/dL)", min_value=70, max_value=300, value=st.session_state.get('blood_sugar', 100), key='blood_sugar')
            
        with col2:
            st.subheader("Lifestyle & Health")
            smoking_options = ["Never", "Former", "Current"]
            smoking_index = smoking_options.index(st.session_state.get('smoking', 'Never'))
            smoking = st.selectbox("Smoking Status", smoking_options, index=smoking_index, key='smoking')
            
            alcohol_options = ["None", "Moderate", "Heavy"]
            alcohol_index = alcohol_options.index(st.session_state.get('alcohol_intake', 'None'))
            alcohol_intake = st.selectbox("Alcohol Intake", alcohol_options, index=alcohol_index, key='alcohol_intake')
            
            exercise_hours = st.slider("Exercise Hours per Week", min_value=0, max_value=20, value=st.session_state.get('exercise_hours', 5), key='exercise_hours')
            stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=st.session_state.get('stress_level', 5), key='stress_level')
            
            st.subheader("Medical History")
            family_history_options = ["No", "Yes"]
            family_index = family_history_options.index(st.session_state.get('family_history', 'No'))
            family_history = st.selectbox("Family History of Heart Disease", family_history_options, index=family_index, key='family_history')
            
            diabetes_options = ["No", "Yes"]
            diabetes_index = diabetes_options.index(st.session_state.get('diabetes', 'No'))
            diabetes = st.selectbox("Diabetes", diabetes_options, index=diabetes_index, key='diabetes')
            
            obesity_options = ["No", "Yes"]
            obesity_index = obesity_options.index(st.session_state.get('obesity', 'No'))
            obesity = st.selectbox("Obesity", obesity_options, index=obesity_index, key='obesity')
            
            exercise_angina_options = ["No", "Yes"]
            exercise_index = exercise_angina_options.index(st.session_state.get('exercise_angina', 'No'))
            exercise_angina = st.selectbox("Exercise Induced Angina", exercise_angina_options, index=exercise_index, key='exercise_angina')
            
            chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            chest_pain_index = chest_pain_options.index(st.session_state.get('chest_pain_type', 'Asymptomatic'))
            chest_pain_type = st.selectbox("Chest Pain Type", chest_pain_options, index=chest_pain_index, key='chest_pain_type')
        
        # Prediction button
        submitted = st.form_submit_button("🔮 Predict Heart Disease Risk", type="primary")

    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Cholesterol': [cholesterol],
            'Blood Pressure': [blood_pressure],
            'Heart Rate': [heart_rate],
            'Smoking': [smoking],
            'Alcohol Intake': [alcohol_intake],
            'Exercise Hours': [exercise_hours],
            'Family History': [family_history],
            'Diabetes': [diabetes],
            'Obesity': [obesity],
            'Stress Level': [stress_level],
            'Blood Sugar': [blood_sugar],
            'Exercise Induced Angina': [exercise_angina],
            'Chest Pain Type': [chest_pain_type]
        })
        
        try:
            # Preprocess the input data to match training features
            processed_data = preprocess_prediction_input(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Calculate percentages
            no_disease_prob = prediction_proba[0] * 100
            disease_prob = prediction_proba[1] * 100
            confidence = max(prediction_proba) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"🚨 **HIGH RISK** ({disease_prob:.1f}%)")
                    st.write("The model predicts a high likelihood of heart disease.")
                else:
                    st.success(f"✅ **LOW RISK** ({disease_prob:.1f}%)")
                    st.write("The model predicts a low likelihood of heart disease.")
            
            with col2:
                # Create a simple gauge-like chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = disease_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Heart Disease Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if prediction == 1 else "green"},
                        'steps' : [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightcoral"}],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            
            # Risk factors analysis
            st.subheader("📊 Key Risk Factors Analysis")
            
            risk_factors = []
            if age > 60: risk_factors.append(f"Advanced Age ({age})")
            if cholesterol > 240: risk_factors.append(f"High Cholesterol ({cholesterol})")
            if blood_pressure > 140: risk_factors.append(f"High Blood Pressure ({blood_pressure})")
            if smoking == "Current": risk_factors.append("Current Smoking")
            if diabetes == "Yes": risk_factors.append("Diabetes")
            if obesity == "Yes": risk_factors.append("Obesity")
            if family_history == "Yes": risk_factors.append("Family History")
            if stress_level > 7: risk_factors.append(f"High Stress Level ({stress_level})")
            if exercise_angina == "Yes": risk_factors.append("Exercise Induced Angina")
            if chest_pain_type == "Typical Angina": risk_factors.append("Typical Angina")

            
            if risk_factors:
                st.warning(f"**Identified Risk Factors based on your input:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.info("**No major risk factors identified from your input.**")
            
            # Recommendations
            st.subheader("💡 General Recommendations")
            if prediction == 1:
                st.error("""
                **Based on this HIGH-RISK prediction, please consider these general recommendations:**
                - **Consult a Doctor:** This is an AI prediction, not a medical diagnosis. Please see a healthcare professional for a proper evaluation.
                - **Monitor Vitals:** Regularly check your blood pressure, cholesterol, and blood sugar.
                - **Lifestyle Review:** Discuss diet, exercise, and stress management with your doctor.
                """)
            else:
                st.success("""
                **Based on this LOW-RISK prediction, it's great to continue:**
                - **Maintain Healthy Lifestyle:** Continue regular exercise and a balanced diet.
                - **Regular Checkups:** Don't skip your annual health checkups.
                - **Stress Management:** Continue to manage stress levels effectively.
                """)
            
            st.caption("Disclaimer: This is an AI model and not a substitute for professional medical advice, diagnosis, or treatment.")

                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("There might be an issue with the preprocessing steps or the loaded model.")
            st.info("Please ensure your input values are correct and the model's expected features match the 'preprocess_prediction_input' function.")
            import traceback
            st.error(traceback.format_exc())


# --- NEW FUNCTION FOR THE CHATBOT ---

def show_chatbot_page(df):
    st.header("❤️‍🩹 Heart Health AI Assistant")
    st.info("Ask me questions about the heart disease data. For example: 'What's the average cholesterol for patients with heart disease?' or 'How many smokers also have diabetes?'")

    # Configure the Gemini API
    try:
        # Try to get the key from st.secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except (KeyError, FileNotFoundError):
        st.error("Error: GOOGLE_API_KEY not found.")
        st.info("Please add your Google API Key to a file named .streamlit/secrets.toml")
        st.code("GOOGLE_API_KEY = \"YOUR_API_KEY_HERE\"")
        return # Stop the function if key is missing
    except Exception as e:
        st.error(f"Error configuring Google AI: {e}")
        return

    # Set up the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze the heart data today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create the LLM and Agent
    # We cache this to prevent re-creating it on every app rerun
    @st.cache_resource
    def get_agent(_df):
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        
        # This is the prompt that guides the AI
        agent_prompt = f"""
        You are a friendly and helpful data analysis assistant.
        You are working with a pandas DataFrame in Python. The name of the dataframe is `df`.
        The dataframe contains a heart disease dataset.
        
        The columns and their meanings are:
        {df.info()}
        
        Your goals are:
        1.  Answer the user's questions about the data.
        2.  Provide suggestions for a healthy lifestyle based on the data insights.
        3.  Be conversational and clear in your responses.
        4.  If asked for a plot, you must use plotly.express. Do not use matplotlib or seaborn.
        5.  When asked a question, you must first think about what to do.
        
        When answering, first perform the necessary pandas operations (if any), get the result, 
        and then formulate a user-friendly response.
        
        IMPORTANT: When asked to plot, generate the Python code for a plotly express chart and 
        return it in the 'output'. Do not attempt to show the plot yourself.
        
        Example:
        User: "What is the average age?"
        Thought: I need to calculate the mean of the 'Age' column.
        Action: python_repl_ast("print(df['Age'].mean())")
        Observation: 52.29
        Thought: The average age is 52.29. I will tell this to the user.
        Final Answer: The average age of the patients in the dataset is 52.29 years.
        """
        
        # This creates the agent
        try:
            agent = create_pandas_dataframe_agent(
                llm,
                _df,
                prompt=agent_prompt,
                verbose=True, # Set to True to see the agent's thought process in the console
                allow_dangerous_code=True # Required to execute pandas code
            )
            return agent
        except Exception as e:
            st.error(f"Error creating agent: {e}")
            return None

    agent = get_agent(df)
    if agent is None:
        return # Stop if agent creation failed

    # React to user input
    if prompt := st.chat_input("What is the average age of patients?"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use the agent to invoke a response
                    response = agent.invoke(prompt)
                    response_content = response['output']
                except Exception as e:
                    response_content = f"Sorry, I ran into an error: {e}"
                
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})


if __name__ == "__main__":
    main()