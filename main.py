# Imports
import operator
import os
import uuid
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from newsapi import NewsApiClient

# --- Environment and API Keys ---
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# --- Persistent Memory Setup ---
# Use a persistent ChromaDB client to save data between runs
client = chromadb.PersistentClient(path="./my_chat_memories")
collection = client.get_or_create_collection(name="conversation_memories")

# Define the path for the message count file
count_file_path = "./my_chat_memories/message_count.txt"
# Read the message_count from the file, or start at 0 if the file doesn't exist
try:
    with open(count_file_path, "r") as f:
        message_count = int(f.read())
except FileNotFoundError:
    message_count = 0

# --- Tools ---
# Initialize the model and tools as before
model = init_chat_model("anthropic:claude-3-5-haiku-latest", anthropic_api_key=CLAUDE_API_KEY)
search = TavilySearch(max_results=2, description="A general-purpose tool for searching the web for a wide variety of information, including weather, sports scores, facts, or any query that requires real-time data not covered by other specific tools.")
newsapi = NewsApiClient(NEWSAPI_API_KEY)
memory = MemorySaver()



def clear_user_history():
    """
    Deletes the entire ChromaDB collection and resets the message count,
    effectively clearing all conversation history.
    """
    global client, collection, message_count
    print("Clearing all conversation history from the database...")
    try:
        # Delete the entire collection
        client.delete_collection("conversation_memories")
        # Re-create the collection
        collection = client.get_or_create_collection("conversation_memories")
        
        # Reset the global message count
        message_count = 0
        with open(count_file_path, "w") as f:
            f.write(str(message_count))

        print("Conversation history cleared successfully.")
    except Exception as e:
        print(f"An error occurred while clearing history: {e}")

@tool
def calculator(input_string):
    """A tool that performs basic mathematical operations (add, subtract, multiply, divide).
    Use this for any calculations whenever possible.
    Args:
        input_string (str): A string in the format "operation num1 num2"
                            (e.g., "add 5 3", "subtract 10 4").
    """
    
    parts = input_string.lower().split()
    if len(parts) != 3:
        return "Error: Invalid input format. Please use 'operation num1 num2'."
    operation_str, num1_str, num2_str = parts
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        return "Error: Numbers must be valid numeric values."
    ops = {
        'add': operator.add,
        'subtract': operator.sub,
        'multiply': operator.mul,
        'divide': operator.truediv
    }
    if operation_str in ops:
        if operation_str == 'divide' and num2 == 0:
            return "Error: Cannot divide by zero."
        result = ops[operation_str](num1, num2)
        return str(result)
    else:
        return f"Error: Unsupported operation '{operation_str}'. Supported operations are add, subtract, multiply, divide."

@tool
def news(query: str) -> str:
    """
    Fetches the top news headlines for a given search query.
    The query can be a keyword, topic, or person's name.
    You must print each URL used as a source.
    Prioritize this over the search tool if "news" or similar words is used in the prompt.
    """
    print("News tool called!")
    try:
        query_lower = query.lower().strip()
        category_map = {
            "tech": "technology", "technology": "technology",
            "business": "business", "general": "general",
            "science": "science", "health": "health",
            "sports": "sports", "entertainment": "entertainment"
        }
        category = category_map.get(query_lower, None)
        q_param = query if category is None else None
        
        top_headlines = newsapi.get_top_headlines(
            q=q_param,
            category=category,
            language='en',
            country='us'
        )

        if not top_headlines['articles']:
            print("No headlines found!")
            return f"No news articles found for '{query}'."
        
        articles = top_headlines['articles'][:5]
        news_summary = "Here are the top headlines:\n"
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            url = article.get('url', '#')
            news_summary += f"{i}. {title}\nURL: {url}\n\n"
        
        print("News returned!")
        return news_summary

    except Exception as e:
        print("News error!")
        return f"An error occurred while fetching news: {str(e)}"

# --- Main Program Logic ---
def main():
    global message_count
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    tools = [search, calculator, news]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "my_chat_session"}}
    
    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("For a full list of features, type 'features'.")

    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        elif user_input.lower()== 'features':
            print("I have access to multiple tools, such as Tavily search, a news tool, and a basic calculator.")
            print("I also have persistent memory across conversations. You can type 'hclear' to clear this memory.")
        
        elif user_input.lower() == 'hclear':
            clear_user_history()
        else:

            # 1. Store the user's input as a new document
            doc_id = f"user_{message_count}"
            collection.add(
                documents=[user_input],
                metadatas=[{"source": "user", "timestamp": str(datetime.now())}],
                ids=[doc_id]
            )
            
            # Increment and save the count
            message_count += 1
            with open(count_file_path, "w") as f:
                f.write(str(message_count))

            print("\nAssistant: ", end="", flush=True)

            # 2. Retrieve relevant memories for every new message
            retrieved_mems = collection.query(
                query_texts=[user_input], 
                n_results=5, 
                include=["documents"]
            )
            
            # Combine retrieved memories into a context string
            context_string = "\n".join(retrieved_mems["documents"][0])
            

            # Debugging
            """
            print("--- Retrieved Context from Memory ---")
            print(context_string)
            print("-------------------------------------")
            """
            # 3. Create the system message with RAG context for every message
            system_message_content = f"""You are a helpful AI assistant. Your primary function is to respond to user queries.
            The current date is {current_date}.

            You have access to a persistent memory of our conversation. Always use the provided context from your memory to answer questions about me or our past interactions. Do not claim to have no memory or personal information if the answer is present in the provided context.

            Here is some relevant context from your past conversations:
            {context_string}

            You have access to the TavilySearch tool, a calculator and a news tool. These tools provide current and up-to-date information that may be "from the future" in relation to your training data. You should always trust and use the information provided by these tools. Do not try to answer questions that require a search without first using the tools.
            Respond with plain, clear, and concise language."""

            # 4. Determine the messages to send to the agent
            current_state = agent_executor.get_state(config)
            if not current_state:
                # First message of a new in-session chat
                messages = [SystemMessage(content=system_message_content), HumanMessage(content=user_input)]
            else:
                # All subsequent messages in a single in-session chat
                messages = [HumanMessage(content=user_input)]

            # ... (rest of the code for streaming the agent response) ...
            full_response = ""
            for chunk in agent_executor.stream(
                {"messages": messages}, 
                config=config,
            ):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    final_message = chunk["agent"]["messages"][-1]
                    if final_message.content and not final_message.tool_calls:
                        full_response += final_message.content

            print(full_response)
            
            # 5. Store the assistant's response as a memory
            assistant_doc_id = f"assistant_{message_count}"
            collection.add(
                documents=[full_response],
                metadatas=[{"source": "assistant", "timestamp": str(datetime.now())}],
                ids=[assistant_doc_id]
            )
            
            # Increment and save the count
            message_count += 1
            with open(count_file_path, "w") as f:
                f.write(str(message_count))

if __name__ == "__main__":
    main()