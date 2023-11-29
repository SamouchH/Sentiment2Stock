import streamlit as st
import plotly.express as px
import pandas as pd


tweets_sentiment_df = pd.read_csv('./data/stock_tweets_with_sentiment.csv')

def categorize_sentiment(score):
    if score < 0:
        return 'Negative'
    elif score > 0:
        return 'Positive'
    else:
        return 'Neutral'

tweets_sentiment_df['Sentiment Category'] = tweets_sentiment_df['compound_score'].apply(categorize_sentiment)


fig = px.histogram(tweets_sentiment_df, x='compound_score', color='Sentiment Category', title= 'Distribution of Sentiment Scores')



def app():
    st.title('Welcome to Our Stock Analysis Platform')

    st.write("""
        This platform leverages advanced data analysis techniques to provide insights into stock market sentiment 
        and predictive analytics. Navigate through the app using the sidebar to access various features.
    """)

    st.header('How to Use This Platform')
    st.write("""
        - **Sentiment Analysis**: Analyze the sentiment of stock-related news and tweets.
        - **Visualize**: Explore visualizations of stock prices and sentiment trends over time.
        - **Predict**: Utilize our predictive models to forecast future stock prices.
        - **Future Work**: Learn about the upcoming features and enhancements planned for this platform.
    """)

    st.header('A Comprehensive Analysis Tool')
    st.write("""
        Our platform offers multiple tools for investors and analysts to gain a comprehensive 
        understanding of market sentiment and make data-driven decisions.
    """)

    # Assuming 'fig' is a Plotly chart that you have previously defined.
    # Make sure to define 'fig' in this script or import it from another module if needed.
    st.header('Visualizing Market Sentiment')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Powered by Cutting-Edge Technology')
    st.write("""
        At the heart of our platform is a combination of natural language processing and machine learning techniques,
        including the BERT model for sentiment analysis and GANs for stock price prediction.
    """)

    st.header('Get Involved')
    st.write("""
        - **Feedback**: Your feedback is crucial to improving our platform.
        - **Contribute**: Interested in contributing? Check out our Github repository or reach out to us directly.
        - **Stay Updated**: Visit the 'Future Work' page to learn about what's coming next.
    """)

    st.header('About Us')
    st.write('This platform was developed by a dedicated team focused on leveraging technology to provide actionable market insights.')
    st.markdown('Feel free to reach out through [GitHub](https://github.com/yourusername/your-repo).')

    st.sidebar.title('Navigation')
    st.sidebar.info('Use the sidebar to navigate to different pages of the application.')
