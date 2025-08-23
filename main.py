# Imports

import operator
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import tool
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

from datetime import datetime

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")



model = init_chat_model("anthropic:claude-3-5-haiku-latest", anthropic_api_key=CLAUDE_API_KEY)
search = TavilySearch(max_results=2, description="A tool for searching the web for current, up-to-date information like weather, sports scores, or news.")
newsapi = NewsApiClient(NEWSAPI_API_KEY)
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
    Make sure to print the URLS given!
    """
    
    try:
        # Fetch the top headlines for the given query in English, from the US
        top_headlines = newsapi.get_top_headlines(q=query, language='en', country='us')

        if not top_headlines['articles']:
            return f"No news articles found for '{query}'."
        
        # Format the top 5 articles for the agent to read and use
        articles = top_headlines['articles'][:5]
        news_summary = "Here are the top headlines:\n"
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            url = article.get('url', '#')
            news_summary += f"{i}. {title}\nURL: {url}\n\n"

        return news_summary

    except Exception as e:
        return f"An error occurred while fetching news: {str(e)}"

def main():
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    tools = [search, calculator, news]
    agent_executor = create_react_agent(model, tools)

    config = {"configurable": {"thread_id": "my_chat_session"}}
    
    system_message_content = f"""You are a helpful AI assistant. Your primary function is to respond to user queries. The current date is {current_date}.
You have access to the TavilySearch tool, a calculator and a news tool. These tools provide current and up-to-date information that may be "from the future" in relation to your training data. You should always trust and use the information provided by these tools. Do not try to answer questions that require a search without first using the tools.
Respond with plain, clear, and concise language."""
    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform searches or chat with me.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            break
        
        print("\nAssistant: ", end="", flush=True)

        current_state = agent_executor.get_state(config)
        
        if not current_state:
            messages = [SystemMessage(content=system_message_content), HumanMessage(content=user_input)]
        else:
            messages = [HumanMessage(content=user_input)]

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

if __name__ == "__main__":
    main()