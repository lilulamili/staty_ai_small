import streamlit as st
import streamlit.components.v1 as components
import use_case_home
import use_case_Manage_LLMs
import use_case_basic_chat
import use_case_FAQs

#----------------------------------------------------------------------------------------------

###############
# PAGE CONFIG #
###############

# Define page setting
st.set_page_config(
    layout="wide", 
    page_title="STATY.AI", 
    page_icon='default_data/logo.png',
    initial_sidebar_state="expanded",
    )
    
change_header_style = """
    <style>
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stToolbarActions"] { display: none !important; }
    </style>
    """
#st.markdown(change_header_style, unsafe_allow_html=True)

change_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}   
"""
st.markdown(change_footer_style, unsafe_allow_html=True)

change_expander_style = """
    <style>
[data-testid="stExpander"] details {
    #border-style: none; 
    border-top: 0px;
    border-left: 0px;
    border-right: 0px;
    }
</style>
    """
st.markdown(change_expander_style, unsafe_allow_html=True)

# Title
components.html("""
    <body 
    style = "margin-left: 0px;">

        <h1 
        style = "color:#38bcf0;
        font-family: sans-serif;
        font-weight: 750;
        font-size: 40px;
        line-height: 0;"
        >STATY.AI
        </h1> 
        <h2 
        style = "color:#38bcf0;
        font-family: sans-serif;
        font-weight: 200;
        font-size: 16px;
        line-height: 1;"
        >Your personal AI assistant
        </h2>   

    </body>

    """,
    height = 70
)




st.sidebar.title("Menu")


#----------------------------------------------------------------------------------------------

#######
# APP #
#######

# Run different code for different use case
PAGES = {
    "Home": use_case_home,
    "Manage LLMs":use_case_Manage_LLMs,
    "Chat": use_case_basic_chat,   
    "FAQs": use_case_FAQs,
    }
#st.sidebar.subheader("Navigation")
main_navigation = st.sidebar.radio("Navigation", ["Home", "Manage LLMs","Chat", "Chat with my data", "FAQs"])
st.sidebar.markdown("")

if main_navigation=="Home":
    use_case="Home"
elif  main_navigation=="Manage LLMs":
    use_case="Manage LLMs"  
elif main_navigation =="Chat":      
    use_case="Chat"
elif main_navigation=="FAQs":
    use_case="FAQs"

st.session_state['main_navigation'] = main_navigation

page = PAGES[use_case]
page.app()

  

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("Report a [bug](mailto:staty@quant-works.de?subject=Staty.AI-bug)")
st.sidebar.markdown("Your :rocket: to data science!   \n Licensed under the [Apache License, Version 2.0](%s)" % "https://www.apache.org/licenses/LICENSE-2.0.html")
 

# Hide footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""

# Decoration bar color
decoration_bar_style ="""
    <style>
    div[data-testid*="stDecoration"] {
        background-image:  linear-gradient(90deg, #38bcf0, #dcf3fa);
    }
    </style>
"""
st.markdown(decoration_bar_style, unsafe_allow_html=True) 

# Progress bar color
progress_bar_style ="""
    <style>
    .stProgress > div > div > div > div {
        background-color: #38bcf0;
    }
    </style>
"""
st.markdown(progress_bar_style, unsafe_allow_html=True) 


# st.info color
info_container_style ="""
    <style>
    .stAlert .st-al {
        background-color: rgba(220, 243, 252, 0.4);
        color: #262730;
    }
    </style>
"""
st.markdown(info_container_style, unsafe_allow_html=True) 


