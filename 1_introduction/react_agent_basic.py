from langchain_mistralai import ChatMistralAI
from  dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# llm = ChatMistralAI(
#     model="mistral-large-latest",
# )

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro"
# )

llm = ChatNVIDIA(
    model = "deepseek-ai/deepseek-r1"
)

# search_tool = TavilySearchResults(search_depth="basic")
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current system time in the specified format. """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [DuckDuckGoSearchResults(), get_system_time]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)

agent.invoke("When was Spacex's lasr launch and how many days ago was it from this instant")

# result = llm.invoke('Give me a fact about Cats')

# print(result.content)