import streamlit as st
import pandas as pd
import llm_functions as lfc

def app():
    
    #----------------------------------------------------------------
    # Chat with LLMs 
    #----------------------------------------------------------------  
    
    #avatar logos
    ai_avatar='default_data/logo.png'
    hs_avatar = 'default_data/hs_logo1.png'

    if 'promt' not in st.session_state:
        st.session_state['promt']="Chat with STATY.AI"

    # settings sidebar
    (sett_theme,output_format)=lfc.settings_sidebar()

    #model selection sidebar
    (menu_option, available_ollama_models,llm_name)=lfc.select_model_sidebar() 
                    
    # model paramater settings sidebar       
    if llm_name:
        # conversation key  - model & menu_option
        conversation_key = f"model_{llm_name}"+menu_option.replace(" ", "")
        
        (model_temp,system_promt)=lfc.model_settings(available_ollama_models,llm_name)
        
        if llm_name=="sroecker/sauerkrautlm-7b-hero:latest":
            prompt = st.chat_input(f"Ask sauerkrautlm-7b-hero a question ...")
        else:
            prompt = st.chat_input(f"Ask '{llm_name}' a question ...")

        
        if conversation_key not in st.session_state.keys():
            st.session_state[conversation_key] = []

        lfc.print_chat_history_timeline(conversation_key)
        
        # run the model
        if prompt:
            st.session_state['promt']=prompt
            st.session_state[conversation_key].append({
                "content": f"{prompt}",#+ system_promt, 
                "role": "user",
                "options": {
                    "seed": 42,
                    "temperature":model_temp
                }})
            with st.chat_message("question", avatar=hs_avatar):
                st.markdown(prompt)

            messages = [dict(
                content=message["content"] +system_promt, 
                role=message["role"], 
                options=message["options"]) 
                    for message in st.session_state[conversation_key]]
                

            # streaming response
            with st.chat_message("response", avatar=ai_avatar):
                #with st.spinner("Generating answer..."):                    
                response_message = st.write_stream(lfc.llm_stream(llm_name, messages))

            st.session_state[conversation_key].append({
                "content": f"{response_message}", 
                "role": "assistant",
                "options": {
                    "seed": 42,
                    "temperature": model_temp
                }})
                
        
        if st.session_state[conversation_key]:            
            # Give a prefix to chat-file name
            file_prefix=st.sidebar.text_input("Give a chat some name", value=lfc.get_chat_name_placeholder(st.session_state['promt']),max_chars=20)
            
            # save conversation to file 
            a4,a5=st.sidebar.columns(2)  
            with a4:         
                save_chat =st.button("Save chat",help="Your conversation will be saved to the folder: 'chat_history'")

            with a5:
                #Clear chat 
                clear_chat = st.button("Clear chat")
                if clear_chat:
                    st.session_state[conversation_key] = []
                    st.rerun()                         
                
            if save_chat:
                    if file_prefix:
                        try:
                            if output_format==".txt":                    
                                lfc.save_chat_to_txt(file_prefix,llm_name, conversation_key)                    
                            elif output_format==".json":                    
                                lfc.save_chat_to_json(file_prefix,llm_name, conversation_key)
                        except Exception as e:
                            st.sidebar.error(f"An error occurred while saving conversation: {str(e)}")
                    else:
                        st.sidebar.error("Please give a name to your chat")
            
         