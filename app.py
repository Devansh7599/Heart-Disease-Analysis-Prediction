import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Add these imports for the chatbot
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import warnings
warnings.filterwarnings('ignore')

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
    df = pd.read_csv('heart_disease_dataset.csv')
    return df

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('best_heart_disease_model.joblib')
    return model

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
    binary_encodings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Smoking': {'Never': 0, 'Former': 1, 'Current': 2},
        'Diabetes': {'No': 0, 'Yes': 1},
        'Obesity': {'No': 0, 'Yes': 1},
        'Family History': {'No': 0, 'Yes': 1},
        'Exercise Induced Angina': {'No': 0, 'Yes': 1}
    }
    
    for col, encoding in binary_encodings.items():
        if col in df.columns:
            df[col] = df[col].map(encoding)
    
    # One-hot encode categorical features with drop_first=True
    df = pd.get_dummies(df, columns=['Alcohol Intake', 'Chest Pain Type'], drop_first=True)
    
    # Ensure all expected features are present (add missing ones with 0)
    # Based on training data analysis, Heavy was dropped as reference category
    expected_features = [
        'Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 
        'Smoking', 'Exercise Hours', 'Family History', 'Diabetes', 'Obesity', 
        'Stress Level', 'Blood Sugar', 'Exercise Induced Angina',
        'Alcohol Intake_Moderate',  # Heavy is the reference category (dropped)
        'Chest Pain Type_Atypical Angina', 'Chest Pain Type_Non-anginal Pain', 
        'Chest Pain Type_Typical Angina'
    ]
    
    # Add missing columns with 0 values
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the expected features in the correct order
    df = df[expected_features]
    
    # Scale numerical features using the same scaling as training
    numerical_features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level', 'Blood Sugar']
    
    # Use the exact scaling parameters from the training data
    scaling_params = {
        'Age': {'mean': 52.29, 'std': 15.73},
        'Cholesterol': {'mean': 249.94, 'std': 57.91},
        'Blood Pressure': {'mean': 135.28, 'std': 26.39},
        'Heart Rate': {'mean': 79.20, 'std': 11.49},
        'Stress Level': {'mean': 5.65, 'std': 2.83},
        'Blood Sugar': {'mean': 134.94, 'std': 36.70}
    }
    
    for feature in numerical_features:
        if feature in df.columns:
            mean_val = scaling_params[feature]['mean']
            std_val = scaling_params[feature]['std']
            df[feature] = (df[feature] - mean_val) / std_val
    
    return df

# Main app
def main():
    st.title("❤️ Heart Disease Analysis & Prediction Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    model = load_model()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a section:", [
                           "📊 Data Overview & Summary",
                           "📈 Data Visualizations",
                           "🔮 Heart Disease Prediction",
                           "🤖 AI Health Assistant"])
    
    if page == "📊 Data Overview & Summary":
        show_data_overview(df)
    elif page == "📈 Data Visualizations":
        show_visualizations(df)
    elif page == "🔮 Heart Disease Prediction":
        show_prediction(df, model)
    elif page == "🤖 AI Health Assistant": # New page logic
        show_chatbot(df)

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
        st.write(missing_data[missing_data > 0])
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Categorical variables summary
    st.subheader("Categorical Variables Summary")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        st.write(f"**{col}:**")
        st.write(df[col].value_counts())
        st.write("---")

def show_visualizations(df):
    st.header("📈 Data Visualizations")
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Distribution Charts", 
        "📈 Correlation Heatmap", 
        "📉 Box Plots", 
        "🔗 Pair Plot", 
        "📊 Count Plots", 
        "📈 Scatter Plots", 
        "📊 Target Analysis"
    ])
    
    with tab1:
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
    
    with tab2:
        st.subheader("Correlation Heatmap")
        
        # Calculate correlation for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap of Numerical Features"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
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
    
    with tab4:
        st.subheader("Pair Plot Analysis")
        
        # Select subset of features for pair plot
        selected_features = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Heart Disease']
        
        # Create scatter matrix with enhanced visibility
        fig = px.scatter_matrix(
            df[selected_features],
            dimensions=selected_features[:-1],
            color='Heart Disease',
            title="Pair Plot Analysis of Key Features",
            color_discrete_map={0: '#00FF7F', 1: '#FF1493'},  # Bright green and hot pink for better contrast
            symbol='Heart Disease',
            symbol_map={0: 'circle', 1: 'diamond'},
            opacity=0.8,  # Increased opacity for better visibility
            size_max=8,   # Larger point size
            labels={'Heart Disease': 'Heart Disease Status'},
            width=800,    # Fixed width for consistency
            height=800    # Fixed height for consistency
        )
        
        # Update layout for enhanced visibility
        fig.update_layout(
            height=800,
            font=dict(size=14, family="Arial", color='white'),
            title_font=dict(size=18, family="Arial", color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.95)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=2,
                font=dict(size=13, color='white')
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            margin=dict(l=50, r=50, t=80, b=80)
        )
        
        # Update axes styling for better visibility
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=False,
            linecolor='white',
            linewidth=2,
            tickfont=dict(size=12, color='white'),
            title_font=dict(size=13, color='white')
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=False,
            linecolor='white',
            linewidth=2,
            tickfont=dict(size=12, color='white'),
            title_font=dict(size=13, color='white')
        )
        
        # Add enhanced annotations for better understanding
        fig.add_annotation(
            text="<b>📊 Legend:</b> <span style='color:#00FF7F'>● Green circles</span> = No Heart Disease | <span style='color:#FF1493'>◆ Pink diamonds</span> = Heart Disease",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=14, color='white', family="Arial"),
            align="center",
            bgcolor="rgba(0,0,0,0.9)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Count Plots for Categorical Variables")
        
        categorical_cols = ['Gender', 'Smoking', 'Diabetes', 'Obesity']
        
        for col in categorical_cols:
            fig = px.histogram(
                df, 
                x=col, 
                color='Heart Disease',
                title=f"Distribution of {col} by Heart Disease Status",
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
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
    
    with tab7:
        st.subheader("Target Variable Analysis")
        
        # Heart Disease distribution
        fig1 = px.pie(
            df, 
            names='Heart Disease', 
            title="Distribution of Heart Disease Cases",
            color_discrete_map={0: 'lightblue', 1: 'red'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Heart Disease by Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 40, 60, 80, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        fig2 = px.histogram(
            df, 
            x='Age_Group', 
            color='Heart Disease',
            title="Heart Disease Distribution by Age Groups",
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_prediction(df, model):
    st.header("🔮 Heart Disease Prediction")
    
    # Show example values for low risk
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.slider("Age", min_value=18, max_value=100, value=st.session_state.get('age', 50), key='age')
        gender = st.selectbox("Gender", ["Male", "Female"], index=1 if st.session_state.get('gender', 'Female') == 'Female' else 0, key='gender')
        cholesterol = st.slider("Cholesterol", min_value=100, max_value=400, value=st.session_state.get('cholesterol', 200), key='cholesterol')
        blood_pressure = st.slider("Blood Pressure", min_value=80, max_value=200, value=st.session_state.get('blood_pressure', 120), key='blood_pressure')
        heart_rate = st.slider("Heart Rate", min_value=50, max_value=120, value=st.session_state.get('heart_rate', 70), key='heart_rate')
    
    with col2:
        st.subheader("Lifestyle & Health")
        smoking_options = ["Never", "Former", "Current"]
        smoking_index = smoking_options.index(st.session_state.get('smoking', 'Never'))
        smoking = st.selectbox("Smoking Status", smoking_options, index=smoking_index, key='smoking')
        
        alcohol_options = ["None", "Moderate", "Heavy"]
        alcohol_index = alcohol_options.index(st.session_state.get('alcohol_intake', 'None'))
        alcohol_intake = st.selectbox("Alcohol Intake", alcohol_options, index=alcohol_index, key='alcohol_intake')
        
        exercise_hours = st.slider("Exercise Hours per Week", min_value=0, max_value=20, value=st.session_state.get('exercise_hours', 5), key='exercise_hours')
        
        family_history_options = ["No", "Yes"]
        family_index = family_history_options.index(st.session_state.get('family_history', 'No'))
        family_history = st.selectbox("Family History of Heart Disease", family_history_options, index=family_index, key='family_history')
        
        diabetes_options = ["No", "Yes"]
        diabetes_index = diabetes_options.index(st.session_state.get('diabetes', 'No'))
        diabetes = st.selectbox("Diabetes", diabetes_options, index=diabetes_index, key='diabetes')
        
        obesity_options = ["No", "Yes"]
        obesity_index = obesity_options.index(st.session_state.get('obesity', 'No'))
        obesity = st.selectbox("Obesity", obesity_options, index=obesity_index, key='obesity')
        
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=st.session_state.get('stress_level', 5), key='stress_level')
        blood_sugar = st.slider("Blood Sugar", min_value=70, max_value=300, value=st.session_state.get('blood_sugar', 100), key='blood_sugar')
        
        exercise_angina_options = ["No", "Yes"]
        exercise_index = exercise_angina_options.index(st.session_state.get('exercise_angina', 'No'))
        exercise_angina = st.selectbox("Exercise Induced Angina", exercise_angina_options, index=exercise_index, key='exercise_angina')
        
        chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain_index = chest_pain_options.index(st.session_state.get('chest_pain_type', 'Asymptomatic'))
        chest_pain_type = st.selectbox("Chest Pain Type", chest_pain_options, index=chest_pain_index, key='chest_pain_type')
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary"):
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **HIGH RISK** - Heart Disease Predicted")
                    st.error(f"**Risk Level: {disease_prob:.1f}%**")
                else:
                    st.success("✅ **LOW RISK** - No Heart Disease Predicted")
                    st.success(f"**Risk Level: {disease_prob:.1f}%**")
            
            with col2:
                st.metric("No Heart Disease Probability", f"{no_disease_prob:.1f}%")
            
            with col3:
                st.metric("Heart Disease Probability", f"{disease_prob:.1f}%")
            
            # Additional percentage display
            st.markdown("---")
            st.subheader("📊 Detailed Risk Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Breakdown:**")
                st.write(f"• **No Heart Disease:** {no_disease_prob:.1f}%")
                st.write(f"• **Heart Disease:** {disease_prob:.1f}%")
                st.write(f"• **Model Confidence:** {confidence:.1f}%")
            
            with col2:
                # Create a simple progress bar for risk level
                risk_color = "red" if disease_prob > 50 else "green"
                st.write("**Risk Level Visualization:**")
                st.progress(disease_prob / 100)
                st.write(f"Risk Level: {disease_prob:.1f}%")
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            
            risk_factors = []
            if age > 65:
                risk_factors.append("Advanced Age")
            if cholesterol > 240:
                risk_factors.append("High Cholesterol")
            if blood_pressure > 140:
                risk_factors.append("High Blood Pressure")
            if smoking == "Current":
                risk_factors.append("Current Smoking")
            if diabetes == "Yes":
                risk_factors.append("Diabetes")
            if obesity == "Yes":
                risk_factors.append("Obesity")
            if family_history == "Yes":
                risk_factors.append("Family History")
            if stress_level > 7:
                risk_factors.append("High Stress Level")
            
            if risk_factors:
                st.warning(f"**Identified Risk Factors:** {', '.join(risk_factors)}")
            else:
                st.info("**No major risk factors identified**")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.error("""
                **Immediate Actions Recommended:**
                - Consult with a cardiologist
                - Regular monitoring of vital signs
                - Lifestyle modifications (diet, exercise)
                - Medication review if applicable
                """)
            else:
                st.success("""
                **Maintain Healthy Lifestyle:**
                - Continue regular exercise
                - Maintain healthy diet
                - Regular health checkups
                - Stress management
                """)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")

# Helper function to get Google API Key
def get_google_api_key():
    """Get Google API Key from Streamlit secrets or user input."""
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    else:
        return st.sidebar.text_input("Google API Key", key="google_api_key", type="password")

def show_chatbot(df):
    st.header("🤖 AI Health Assistant")
    st.markdown("Ask me questions about the heart disease dataset or general health topics!")

    # Get API Key
    api_key = get_google_api_key()
    if not api_key:
        st.info("Please add your Google API Key to continue.")
        st.stop()

    # Initialize the ChatGoogleGenerativeAI model
    try:
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize the AI model. Please check your API key. Error: {e}")
        st.stop()

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            AIMessage(content="Hello! I'm your AI Health Assistant. How can I help you today?")
        ]

    # Display chat messages
    for msg in st.session_state.chat_messages:
        if isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("human").write(msg.content)

    # Handle user input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_messages.append(HumanMessage(content=prompt))
        st.chat_message("human").write(prompt)

        # Prepare context for the AI
        dataset_summary = df.head().to_string() + "\n\n" + df.describe().to_string()
        full_prompt = f"""
        You are an expert AI health assistant. Your knowledge is based on a heart disease dataset and general medical knowledge.
        
        Here is a summary of the dataset you have access to:
        {dataset_summary}
        
        Based on this context and your general knowledge, please answer the user's question.
        
        User's question: "{prompt}"
        """

        # Get AI response
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke([HumanMessage(content=full_prompt)])
                ai_response = AIMessage(content=response.content)
                st.session_state.chat_messages.append(ai_response)
                st.chat_message("ai").write(ai_response.content)
            except Exception as e:
                st.error(f"An error occurred while getting the response: {e}")

if __name__ == "__main__":
    main()
# https://devansh7599-heart-disease-analysis-prediction-app-1kmh9a.streamlit.app/