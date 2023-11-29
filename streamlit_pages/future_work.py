import streamlit as st

# Define your future work page function
def app():
    st.title('Future Work and Roadmap')

    st.markdown("""
    Here we outline the upcoming features and enhancements planned for this project. 
    Our roadmap is a glimpse into the future of this application, detailing the exciting developments ahead.
    """)

    # If you have specific subsections, you can use headers or subheaders
    st.header("Model Enhancements")
    st.write("We are exploring advanced model architectures and more sophisticated hyperparameter tuning strategies to improve accuracy.")

    st.header("Real-time Stock Price Visualization")
    st.write("A real-time stock price plot will be implemented to provide live financial data visualizations.")

    st.header("Utilizing Reinforcement Learning")
    st.write("We plan to investigate reinforcement learning techniques for better predictive modeling and to inform trading strategies.")

    # To add a screenshot or any image
    st.header("Preview of Upcoming Features")
    st.write("Below is a sneak peek of what we are working on:")
    
    # Assuming you have a path to your screenshot image
    screenshot_path = './images/Capture_1.png'
    st.image(screenshot_path, use_column_width=True)

    screenshot_path = './images/Capture_2.png'
    st.image(screenshot_path, use_column_width=True)


    # Include any additional sections or calls to action
    st.header("Get Involved")
    st.write("Your feedback is invaluable to us. If you have suggestions or would like to contribute, please reach out.")

    st.header("Stay Tuned")

