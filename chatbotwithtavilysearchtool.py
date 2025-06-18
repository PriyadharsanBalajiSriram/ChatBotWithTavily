!pip install -U langchain langgraph langchain-tavily langchain_google_genai
import os
from getpass import getpass
os.environ["GOOGLE_API_KEY"]=getpass("Enter your gemini api key:")
os.environ["TAVILY_API_KEY"]=getpass("Enter your tavily api key:")

!pip install -U langchain_tavily langchain_google_genai
from langchain_tavily import TavilySearch
tool=TavilySearch(max_results=2)
Toolscreated=[tool]
from langchain.chat_models import init_chat_model
llm=init_chat_model("google_genai:gemini-2.0-flash")

!pip install -U langgraph
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
  messages:Annotated[list,add_messages]

workflow=StateGraph(State)

llm_with_tools=llm.bind_tools(Toolscreated)

def toolmakecall(state:State):
  return{
      "messages":[llm_with_tools.invoke(state['messages'])]
  }

workflow.add_node("chatbot",toolmakecall)

import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
workflow.add_node("tools", tool_node)

def routetools(state:State):
  if isinstance(state,list):
    aimessage=state[-1]
  elif messages := state.get("messages", []):
    aimessage = messages[-1]
  else:
    raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(aimessage,"tool_calls") and len(aimessage.tool_calls)>0:
      return "Toolscreated"
  return END

workflow.add_conditional_edges("chatbot",routetools,{"tools":"tools",END:END})

workflow.add_edge(START,"chatbot")
workflow.add_edge("tools","chatbot")
workflow.compile()

