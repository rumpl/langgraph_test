"""test"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from termcolor import colored


@tool
def read_dockerfile():
    """Call to read a Dockerfile."""
    print(colored("read_dockerfile", "cyan"), flush=True)
    # This is a placeholder, but don't tell the LLM that...
    return """
        Read any files that are needed to optimize this Dockerfile:
        
        [DOCKERFILE_START]
        # syntax=docker/dockerfile:1

        FROM python:3.12-alpine3.20@sha256:38e179a0f0436c97ecc76bcd378d7293ab3ee79e4b8c440fdc7113670cb6e204 AS base
        RUN pip --no-cache-dir install poetry==1.8.3

        ENV POETRY_NO_INTERACTION=1 \
            POETRY_VIRTUALENVS_IN_PROJECT=1 \
            POETRY_VIRTUALENVS_CREATE=1 \
            POETRY_CACHE_DIR=/tmp/poetry_cache

        # Install dependencies
        WORKDIR /app
        COPY llm/pyproject.toml llm/poetry.lock ./llm/
        COPY recipes/pyproject.toml recipes/poetry.lock ./recipes/
        COPY server/pyproject.toml server/poetry.lock ./server/
        RUN mkdir -p ./llm/src/llm ./recipes/src/recipes
        RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry -C /app/server install --no-root

        # Copy the sources
        COPY . ./

        # Check the some well knwon files are properly copied
        RUN test -f ./data/run_image/data.json

        # Get the git commit and use it as the version
        ARG GIT_COMMIT="HEAD"
        RUN sed -i "s/HEAD/$GIT_COMMIT/" server/server/settings.py

        FROM base AS test
        RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
            poetry -C /app/tests install --no-root

        RUN --mount=type=secret,id=openai,env=OPENAI_API_KEY \
            poetry -C /app/tests run pytest -n 4 --junitxml=/out/test-results.xml || true

        FROM scratch AS test-results
        COPY --from=test /out/test-results.xml /test-results.xml

        FROM base AS final
        EXPOSE 8000
        ENTRYPOINT ["/app/scripts/entrypoint.sh"]
        [DOCKERFILE_END]
        """


@tool
def read_file(file: str):
    """Call to read a file."""
    print(colored(f"read_file {file}", "cyan"), flush=True)
    # This is a placeholder, but don't tell the LLM that...
    return "module github.com/docker/ai\n\ngo 1.23.2"


@tool
def optimize_dockerfile(dockerfile: str):
    """Call to optimize a Dockerfile."""
    print(colored(f"optimize_dockerfile {dockerfile[:11]}...", "cyan"), flush=True)
    # This is a placeholder, but don't tell the LLM that...
    return f"Here is the optimized Dockerfile. {dockerfile}"


@tool
def write_dockerfile(dockerfile: str):
    """Write a Dockerfile."""
    print(colored(f"write_dockerfile {dockerfile}...", "cyan"), flush=True)
    # This is a placeholder, but don't tell the LLM that...
    return f"Here is the optimized Dockerfile. {dockerfile}"


tools = [read_dockerfile, read_file, optimize_dockerfile, write_dockerfile]

tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0).bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # print(last_message, flush=True)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="Can you optimize my Dockerfile?")]},
    config={"configurable": {"thread_id": 42}},
)
print(final_state["messages"][-1].content)
