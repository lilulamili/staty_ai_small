import streamlit as st
import llm_functions as lfc
import platform

def app():
    
    #----------------------------------------------------------------
    # Chat with LLMs 
    #----------------------------------------------------------------  
    
    # settings sidebar
    #lfc.settings_sidebar()

    #---------------------
    
    st.write("STATY.AI offers a user-friendly interface for downloading and running open access Large Language Models (LLMs) locally on your PC.")    
    #st.markdown("Managing LLMs is done using Ollama (https://ollama.com).") 
    st.markdown("Managing LLMs is done using <a href='https://ollama.com' style='color:#38bcf0'>Ollama</a>.", unsafe_allow_html=True) 

    st.markdown("---") 

    st.header("FAQs")
    st.markdown("If your question is not answered, please [contact us](mailto:staty@quant-works.de?subject=Staty-App)!")
    
    general_questions_container = st.container()
    with general_questions_container:
        st.subheader("General questions")
        
        with st.expander("What LLMs can be used and where are they comming from?"):
            st.write("LLMs are downloaded from Ollama. For full list of available models please check https://ollama.com/library.")
        
        with st.expander("How to download or delete the models?"):
            st.markdown(""" 
                        1) Make sure that Ollama (https://ollama.com/) is downloaded and running!                           
                        2) Select the menu 'Manage LLMs' and then select the appropriate options.""")
        
        
        with st.expander("Where are the models stored on my PC?"):
            os_name = platform.system()
            if os_name=="Windows":            
                st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>/usr/share/ollama/.ollama/models</span>", unsafe_allow_html=True)
            elif os_name=="Darwin":
                st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>~/.ollama/models</span>", unsafe_allow_html=True)
            elif os_name=="Linux":
                st.write("The models will be downloaded to: <span style='background-color: #f2f2f2;'>/usr/share/ollama/.ollama/models</span>", unsafe_allow_html=True)
            
        with st.expander("What are the memory requirements for the LLMs?"):
            st.write("7b models generally require at least 8GB of RAM, 13b models at least 16GB, while 70b models may necessitate up to 64GB of memory.")

        
   
    LLM_container = st.container()
    with LLM_container:
        st.subheader("LLMs")  
        with st.expander("What kind of questions can I ask an LLM in a chat?"):
            st.markdown("LLMs can answer a wide range of open ended, factual, or creative questions. They can provide summaries of factual topics, generate different creative text formats like poems, code, scripts, emails, letters, etc. However, they might struggle with very specific or technical domains requiring deep expertise. ")
         
        with st.expander("What is a session prompt?"):
            st.markdown("A session prompt is a piece of text provided to an LLM that sets the context for the LLM and influences its responses throughout the session.  It can include information about the desired task or specific instructions like- be funny, or provide short answers.")

        with st.expander("How do I select a model temperature?"):
            st.markdown("The model temperature is a parameter that controls the randomness of the LLM's outputs. A higher temperature leads to more creative and surprising, but potentially less accurate responses. A lower temperature results in more predictable and factual answers. The optimal temperature depends on your specific needs and the desired trade-off between creativity and accuracy.")
 
        with st.expander("Should I blindly trust the answers provided by an LLM?"):
            st.markdown("No, you should not blindly trust the answers provided by an LLM. LLMs don't have a true understanding of the world like humans do. They can process information and respond creatively, but they cannot verify the information's truthfulness.")
        
        with st.expander("Are the LLM answers accurate?"):
            st.markdown("LLMs are still under development and might not always provide accurate or complete information. They can be biased based on their training data and may struggle with understanding complex or nuanced questions. It's important to be aware of these limitations and to verify information from other sources when necessary. ")


        with st.expander("Good Practices for Using LLMs Responsibly?"):
            st.markdown("""
                        
                Here are some good practices to consider when using Large Language Models (LLMs):

                * **Be aware of the limitations of LLMs.** LLMs are powerful tools, but they are not perfect. They can be biased, inaccurate, and lack real-world understanding.
                * **Use LLMs as a starting point for research, not a definitive source of truth.** Always verify information from other reliable sources.
                * **Develop critical thinking skills to evaluate the information provided by LLMs.** Don't blindly trust everything an LLM tells you. Look for factual inconsistencies, logical fallacies, and consider the source of the information.
                * **Use LLMs for tasks where creativity or text generation is valuable, not for replacing expert knowledge or verification of critical information.** LLMs are great for generating creative text formats or summarizing factual topics, but they cannot replace the expertise of a human professional.""")
 
 
        
        
        
    data_chat_container = st.container()
    with data_chat_container:
        st.subheader("Chat with your data")
        with st.expander("What is a knowledge base?"):
            st.markdown("""
                        A 'knowledge base' is a collection of structured information used by an LLM to answer questions. 
                        To chat with your own documents, large language models (LLMs) need to understand them first. Regular documents like PDFs or Word files aren't directly interpretable. Therefore, your documents need to be transformed into a format the LLM can understand and stored in a database accessible by LLMs -a 'knowledge base'.
                        """)

        with st.expander(" What is RAG ( Retriever-Augmented Generator )?"):
            st.markdown("RAG is a framework that combines retrieval and generation techniques to allow LLMs to access and process information from a knowledge base. It retrieves relevant documents from the knowledge base based on the user's question and feeds them alongside the question itself to the LLM. This helps the LLM generate more informed and accurate responses.")

        with st.expander("How do I select the number of chunks to use when processing data?"):
            st.markdown("The number of chunks for processing data depends on the size of your data and the capabilities of your system. Smaller chunks are easier to process but might lead to more retrieval steps. Larger chunks can be more efficient but might overwhelm the LLM. Experiment with different chunk sizes to find the optimal balance for your specific use case.")

        with st.expander("How do I select the overlap between chunks?"):
            st.markdown("Overlap between chunks allows the LLM to consider context across boundaries.  A small overlap might lead to information loss at chunk transitions. A large overlap increases processing time.  Start with a small overlap and adjust it based on your data and the quality of the retrieved information.")       

        with st.expander(" What is a splitter function?"):
            st.markdown("A splitter function is responsible for dividing your data into chunks for processing. It defines the criteria used to separate the data. Common splitters might use delimiters like paragraphs, sentences or a specific characters like 'split_here'.")       

        