# STATY.AI

**STATY.AI offers a user-friendly interface for downloading and running open access Large Language Models (LLMs) locally on your PC.**   

**STATY.AI offers two options to get started:**

* **Source Code:** This guide provides detailed instructions for installing and running STATY.AI from the source code. 
* **Portable Version:** Information on obtaining a user-friendly, ready-to-run portable version can be found [here](https://github.com/lilulamili/STATY_AI/wiki/STATY.AI).

> [!TIP]
> * 8GB RAM minimum is recommended. Response speed scales with CPU/GPU power.
> * Expect a slight delay for the first AI response due to initialization.
> * 'Chat with your data' is an experimental feature with best performance on high-end PCs.


## Getting started with source code 

**1. Python Installation**    
> [!TIP]
> Make sure to check the checkbox labeled "Add Python ... to PATH" during the installation process.  
  This ensures you can easily run Python commands from your terminal later.  
   
   Install Python 3.11.8 from: https://www.python.org/downloads/release/python-3118/  
   Windows: Scroll down to the section 'Files' and select `Windows installer (64-bit)`  
   macOS: Scroll down to the section 'Files' and select `macOS 64-bit universal2 installer`

**2. VSC Installation**   
       Install Visual Studio Code by pressing the big blue button from: https://code.visualstudio.com/

**3. Ollama Installation**  
   Install Ollama from: https://ollama.com/

**4. Download 'STATY.AI'**   
  Download 'STATY.AI' as a ZIP file from https://github.com/lilulamili/STATY_AI to a folder on your PC. Click the zip file and extract it.  
  > [!TIP]
> To download STATY.AI, press the green button `<> Code` and select `Download ZIP`. 
      
**5. Get 'STATY.AI' ready**  
   Open VSCode. Go to the `File` menu, select `Open Folder`, and then navigate to the project folder you just created (the one where the file are).  
   `„Do you trust the Author of this Folder” – click “yes”`

**6. create a virtual environment**   
   Locate the `Terminal` panel in VScode (usually at the bottom of the window). If it's not visible, go to the View menu and select Terminal. Type the following command in the terminal to create a virtual environment:

   Windows: `py -3 -m venv .venv`

   macOS:`python3 -m venv .venv `  
   
   `We noticed a new virtual environment....   click "yes"`
   
   > [!TIP]
   > Don't foget the dot before the second venv `.venv`  
> Windows: In case of a policy challenge type the following command in the terminal:   
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**7. Activate the virtual environment**   
  Type the following command in the terminal:

  Windows:`.venv\Scripts\activate`

  macOS:`source .venv/bin/activate` 

**8. Install all application components**   
   Type the following command in the terminal: `pip install -r requirements.txt`

**9. Run STATY.AI**   
   Type the following command in the terminal: `streamlit run staty_ai.py`  
   The app will open in your default browser `http://localhost:8075/`









