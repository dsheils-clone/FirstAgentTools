Tool-utilizing Chatbot built from scratch, utilizing several APIs to enhance functionality.
More tools can easily be added via "tools" list.
  
Setup:  
  uv was used for installation.  
  After cloning the repo, run these commands to install dependencies:  
    pip install uv  
    uv venv  
    source .venv/bin/activate (Linux/Mac) or .venv\Scripts\activate (Windows)  
    uv pip sync requirements.txt  

  To add your own API keys (not included in project):  
    Create a file named .env in the main directory  
    Add your API keys to the .env in this format:  
    CLAUDE_API_KEY=#############  
    
