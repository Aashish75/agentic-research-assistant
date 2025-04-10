# Agentic Research Assistant with LangChain


import os
import arxiv
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 4: Define tools
# -------------------------------
def arxiv_search_tool(query: str) -> str:
    search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in search.results():
        results.append(f"Title: {result.title}\nAbstract: {result.summary}\nLink: {result.entry_id}\n")
    return "\n\n".join(results)

# Wrap tool for LangChain
search_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search_tool,
    description="Useful for searching academic research papers based on a query"
)

# Step 5: Initialize LangChain Agent
# -------------------------------
llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
agents = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 6: Agent Execution
# -------------------------------
def run_agentic_pipeline(query):
    print("\n Running LangChain Agent...")
    response = agents.run(query)
    print("\n Final Output:\n")
    print(response)

# Step 7: Entry point
# -------------------------------
if __name__ == "__main__":
    user_query = input("Enter your research topic or question: ")
    run_agentic_pipeline(user_query)
