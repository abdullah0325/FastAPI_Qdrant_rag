import streamlit as st
import requests

# API Configuration
API_URL = "http://localhost:8000/query"  # FastAPI endpoint

def main():
    st.title("CV Question Answering App")

    # User input
    query = st.text_input("Ask a question about the CV:")

    if query:
        try:
            # Send request to FastAPI backend
            response = requests.post(API_URL, json={"question": query})
            
            if response.status_code == 200:
                # Display the answer
                answer = response.json()['answer']
                st.write(answer)
            else:
                st.error(f"Error: {response.text}")
        
        except requests.RequestException as e:
            st.error(f"Network error: {e}")

if __name__ == "__main__":
    main()