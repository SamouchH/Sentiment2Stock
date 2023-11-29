import streamlit as st
import home
import analysis
import data_insights
import predict
import future_work



#Dictionary mapping page names to their respective modules
pages = {
    "Home": home,
    "Sentiment Analysis": analysis,
    "Visualize": data_insights,
    "Predict": predict,
    "Future Work": future_work
}

#Set page config with light mode and wide mode


#Sidebar for nabigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

#Display the selected page
if selection in pages :
    pages[selection].app()
else:
    st.error("Page Not Found :(")

