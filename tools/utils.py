from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


def get_tool(name, llm=None):
    if name == "Search":
        search = SerpAPIWrapper()
        return Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        )
    elif name == "Calculator":
        assert llm is not None
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        return Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    elif name == "OCR":
        from tools.ocr import ocr
        return Tool(
            name="OCR",
            func=ocr,
            description="perform ocr on an image for Optical Character Recognition."
        )