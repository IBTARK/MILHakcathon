class Assistant:

    def __init__(
        self
    ):
        # Define the system prompt
        self.system_prompt = SYSTEM_PROMPT

        # Define the list tools to be used
        self.tools = [wikipedia_search, web_search, url_markdown, arxiv_search, weather_at_day, weather_now, current_date, current_hour, current_datetime, current_day_of_week, add, subtract, multiply, divide, modulus]

        # Define the model to be used
        llm = ChatOpenAI(model = "gpt-4o", api_key = os.getenv("OPENAI_API_KEY"))
        self.llm_with_tools = llm.bind_tools(self.tools)

        self.graph = self.build_graph()

    # Nodes
    def assistant(
        self,
        state: MessagesState
    ):
        return {"messages": [self.llm_with_tools.invoke([self.system_prompt] + state["messages"])]}

    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(self.tools))

        # Edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        return builder.compile()