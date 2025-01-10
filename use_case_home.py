import streamlit as st
import llm_functions as lfc
import platform
import base64
from PIL import Image
#----------------------------------------------------------------------------------------------

def app():

    #check if Ollama is running
    lfc.ollama_check_home()

    # settings sidebar
    lfc.settings_sidebar()
       
    staty_info="Keep your AI chat private. Open-source AI, made simple.Download and use LLMs locally."
    lfc.typewriter(staty_info, speed=5)      
    st.markdown("")  
   
    
    st.markdown("STATY.AI offers a user-friendly interface for downloading and running open access Large Language Models (LLMs) locally on your PC.")    
   # lfc.typewriter(staty_info, speed=5)  
    st.markdown("Managing LLMs is done using <a href='https://ollama.com' style='color:#38bcf0'>Ollama</a>.", unsafe_allow_html=True) 
    st.markdown("")
    
    
    show_gif=st.toggle("Prefer a short animated guide (GIF) for getting started?", value=False)
    if show_gif:
        file_ = open("default_data/staty_ai.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
       
        image_style = """
        <style>
            img {
                width: calc(50vw - 20px);  /* Adjust padding */
            }
            </style>
            """
        # Display the animated gif        
        st.markdown(f"{image_style}", unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="staty ai">',
            unsafe_allow_html=True,
        )
    else:
        col1, col2 = st.columns([3,2])
        with col1:
            staty_video =open("default_data/staty_ai_intro.mp4", 'rb')        
            staty_video_bytes=staty_video.read()
            st.video(staty_video_bytes)
   
               
        
      
             
    st.markdown("")
    
    #-INFO----------------------------------------------------------------------------------
    staty_expander=st.expander('**STATY.AI - get more info**')    
    with staty_expander: 
        os_name = platform.system()
        st.write("")
        lfc.staty_ai_info()
        if os_name=="Windows":            
            st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>/usr/share/ollama/.ollama/models</span>", unsafe_allow_html=True)
        elif os_name=="Darwin":
            st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>~/.ollama/models</span>", unsafe_allow_html=True)
        elif os_name=="Linux":
            st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>/usr/share/ollama/.ollama/models</span>", unsafe_allow_html=True)
          
        st.write("")
        st.markdown("**Memory requirements**")
        st.write("7b models generally require at least 8GB of RAM, 13b models at least 16GB, while 70b models may necessitate up to 64GB of memory.")

       
        st.write("")
        st.markdown("**Background and motivation**")
        st.markdown("Our overarching goal is to provide students with a comprehensive interdisciplinary understanding and skills to design and apply intelligent models.")

        st.write("")        
        st.markdown("**Disclaimer**")
        st.write("STATY.AI is still under development, and some features may not yet work properly!   \n STATY.AI is provided 'as is' without any warranties of any kind!")
        st.write("")
        st.write("")

    st.markdown("")
    st.markdown("")
    st.write("STATY.AI is an educational project designed and developed with the aim of improving data literacy among students of natural and social sciences.")
    st.write("STATY.AI is provided 'as is' without any warranties of any kind! STATY.AI is under development, and is subject to change!")   
    st.markdown("")
    st.markdown("")
    
    #if sett_theme == "Dark":
    #    image = Image.open("default_data/HS-OS-Logo_dark.png")
   # else:
    #    image = Image.open("default_data/HS-OS-Logo_light.png")
   # with col2:
    #    st.image(image)
  

    
