import streamlit as st
import pandas as pd
import llm_functions as lfc
from PIL import Image


#----------------------------------------------------------------------------------------------

def app():

    # settings sidebar
    lfc.settings_sidebar()

    #---------------------
    
    st.write("STATY.AI offers a user-friendly interface for downloading and running open access Large Language Models (LLMs) locally on your PC.")    
    #st.markdown("Managing LLMs is done using Ollama (https://ollama.com).") 
    st.markdown("Managing LLMs is done using <a href='https://ollama.com' style='color:#38bcf0'>Ollama</a>.", unsafe_allow_html=True) 

    st.markdown("---") 

    #----------------------------------------------------------------------------------
    # Initial Checks
    # check PC configuration
    os_name, total_ram,available_ram,total_disk_space,free_disk_space=lfc.confuguration_check()

    # check available LLMs
    available_ollama_models=lfc.ollama_check()

   
    if available_ollama_models is None:
        model_names=[]    
    else:
        model_names = [model["name"] for model in available_ollama_models]
    
    # Find model proposals       
    (model_proposals,models_dictionary)=lfc.ram_based_models(total_ram)    
    if "sroecker/sauerkrautlm-7b-hero:latest" in model_names:
        model_proposals.remove("sauerkrautlm-7b-hero")
    final_model_suggestion = set(model_proposals) - set(model_names)
    
    #----------------------------------------------------------------------------------
    
    st.subheader("Manage LLMs")  
    st.write("Expand your PC's potential by easily adding or removing LLMs.")     
    
    # Download LLMs ------------------------------------------------------
    download_expander=st.expander("**Download LLMs**",expanded=False)
    with download_expander: 
        a4,a5=st.columns(2)
        with a4: 
            #RAM
            st.pyplot(lfc.create_donut_chart(total_ram, available_ram,fig_size=1,font_size=4,plot_para='RAM'),clear_figure=True,use_container_width=False)        
            
        with a5:
            #Disk Space 
            st.pyplot(lfc.create_donut_chart(total_disk_space, free_disk_space,fig_size=1,font_size=4,plot_para='Disk Space'),clear_figure=True,use_container_width=False)    
        
        
        if final_model_suggestion:
            a4,a5=st.columns(2)
            with a4: 
                pull_model_name=st.selectbox("Based on available RAM, consider installing one of the following models:",list(final_model_suggestion) + ['Enter model name'])
                if pull_model_name=='Enter model name':
                    pull_model_name  =st.text_input("Enter model name:",placeholder="phi:2.7b")          
                
            with a5:
                if pull_model_name in final_model_suggestion:        
                    st.write("")                
                    st.info("**Model info:** \n " + models_dictionary[pull_model_name])
                
            # Model download:                              
            run_download = st.button("Download")
            st.write("")
            if run_download:
                if pull_model_name=="sauerkrautlm-7b-hero":pull_model_name="sroecker/sauerkrautlm-7b-hero" 
                
                try:
                    lfc.pull_model(pull_model_name)
                    st.rerun()    
                except Exception as e:
                    st.error("You have either entered a wrong name, or the server is not responding.")
                        
        
        else:   # no model suggestions 
                st.write("**No model suggestions for your RAM ** ")
       
    
    # DELETE LLMs ------------------------------------------------------
    if model_names:
        delete_expander=st.expander("**Delete LLMs**",expanded=False)
        with delete_expander: 
            a4,a5=st.columns(2)  
            with a4:            
                st.write("**Delete LLMs**")         
                remove_model=st.selectbox(label="Select a model to remove:",options=model_names,)
                # Model delete:
                run_delete = st.button("Delete")         
                if run_delete:
                    lfc.delete_model(remove_model)
                    st.rerun()

            with a5:   
                
                # llm details object
                llm_details = [model for model in available_ollama_models if model["name"] == remove_model][0]
                
                # convert size in llm_details from bytes to GB (human-friendly display)
                if type(llm_details["size"]) != str:
                    llm_details["size"] = f"{round(llm_details['size'] / 1e9, 2)} GB"
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.info("Removing " + remove_model +" would free " + llm_details["size"])

        



    #-INFO----------------------------------------------------------------------------------
    staty_expander=st.expander('**STATY.AI - get more info**')    
    with staty_expander: 

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
  

    
