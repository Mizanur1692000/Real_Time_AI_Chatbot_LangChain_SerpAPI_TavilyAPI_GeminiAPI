import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict
import json
import aiohttp

# LangChain and Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# --- FastAPI Setup ---
app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Gemini Setup ---
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please add it to your .env file.")

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
if not serpapi_api_key:
    raise ValueError("❌ SERPAPI_API_KEY not found. Please add it to your .env file.")

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("❌ TAVILY_API_KEY not found. Please add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # Use a supported model
    google_api_key=gemini_api_key,
    temperature=0.7,
)

# Define the Google search tool
@tool
def google_search(query: str) -> str:
    """Perform a Google search using SerpApi and return the answer or snippet."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_api_key
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "answer_box" in results:
            answer_box = results["answer_box"]
            if "answer" in answer_box:
                return f"Answer: {answer_box['answer']}"
            elif "result" in answer_box:
                return f"Result: {answer_box['result']}"
            elif "snippet" in answer_box:
                return f"Snippet: {answer_box['snippet']}"
            else:
                return json.dumps(answer_box, indent=2)
        elif "organic_results" in results and len(results["organic_results"]) > 0:
            first_result = results["organic_results"][0]
            return f"Title: {first_result.get('title', 'N/A')}\nSnippet: {first_result.get('snippet', 'N/A')}"
        else:
            return "No results found for your query."
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Tavily search tool
@tool
async def tavily_search(query: str) -> str:
    """Perform a search using the Tavily API and return the results."""
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    if results:
                        formatted_results = "\n".join(
                            [f"Title: {res.get('title', 'N/A')}\nContent: {res.get('content', 'N/A')}" for res in results]
                        )
                        return formatted_results
                    else:
                        return "No results found for your query."
                else:
                    return f"Tavily API error: {response.status}"
    except Exception as e:
        return f"An error occurred: {e}"

tools = [google_search, tavily_search]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful AI assistant. Use the google_search or tavily_search tools when you need to look up current information or answer questions requiring web search. Choose the appropriate tool based on the query's complexity, preferring tavily_search for in-depth or specialized queries."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # For debugging; can be removed in production
    handle_parsing_errors=True
)

# In-memory session history
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap with history
conversation_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="output"  # AgentExecutor outputs 'output'
)

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = "default_session"):
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"👤 User ({session_id}): {data}")

            try:
                # Use async invoke
                response = await conversation_with_history.ainvoke(
                    {"input": data},
                    config={"configurable": {"session_id": session_id}}
                )
                bot_response = response["output"]
                print(f"🤖 Bot ({session_id}): {bot_response}")

                await websocket.send_text(bot_response)

            except Exception as e:
                error_msg = f"Chatbot error: {e}"
                print(error_msg)
                await websocket.send_text("⚠️ Oops! Something went wrong.")

    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: {session_id}")