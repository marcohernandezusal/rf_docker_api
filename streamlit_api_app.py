# Create: streamlit_api_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

def main():
    st.title("🌲 Random Forest Predictions (API)")
    st.write("Upload a CSV file to get model predictions via API")
    
    # API configuration
    api_url = st.text_input("API URL:", value="http://127.0.0.1:8000/predict")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head())
            
            if st.button("🔮 Make Predictions", type="primary"):
                predictions = []
                progress_bar = st.progress(0)
                
                # Make predictions row by row (or batch if API supports it)
                for i, row in df.iterrows():
                    try:
                        # Ensure we have exactly 10 features
                        features = row.values[:10].tolist()
                        
                        response = requests.post(
                            api_url,
                            json={"data": features},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            pred = response.json()[0]  # Get first prediction
                            predictions.append(pred)
                        else:
                            st.error(f"API error: {response.status_code}")
                            break
                            
                    except Exception as e:
                        st.error(f"Error in row {i}: {e}")
                        break
                    
                    progress_bar.progress((i + 1) / len(df))
                
                if len(predictions) == len(df):
                    df['predictions'] = predictions
                    st.subheader("📊 Results")
                    st.dataframe(df)
                    
                    # Visualization
                    fig = px.histogram(x=predictions, title="Prediction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        data=csv,
                        file_name="api_predictions.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()