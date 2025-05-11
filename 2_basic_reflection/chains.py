from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a twitter techieinfluencer assistant tasked with writing excellent twitter posts."
         "Generate the best twitter post possible for the user's request."
         " If the user provides critique , respond with the revised version of the previous posts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a viral twitter influencer grading a tweet.Generate critique and recommendations for the user's tweet."
         "Always provide detailed recommendations, including requests perr length, virality , style , etc",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatNVIDIA(
    model = "deepseek-ai/deepseek-r1"
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

 