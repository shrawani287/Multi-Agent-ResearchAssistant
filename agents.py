from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrapr_url
from dotenv import load_dotenv

load_dotenv()

# Model setup (temperature=0 is perfect for maximum factual accuracy)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 1st agent: Specialized Searcher
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a meticulous Web Search Assistant. Your sole job is to discover "
            "highly accurate, recent, and authoritative references on the given topic. "
            "Always use your search tool to find the best available URLs and summaries. "
            "Do not answer from pre-trained knowledge if the user asks for recent information."
        )
    )

# 2nd agent: Specialized Scraper / Content Reader
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrapr_url],
        system_prompt=(
            "You are a deep-dive Document Reading Assistant. Your job is to take the "
            "most promising URL provided from search results and scrape its full contents. "
            "Extract comprehensive data, statistics, and objective information. "
            "Never assume information; rely strictly on what your scraping tool returns."
        )
    )

# writer chain (refined for strict structural and factual adherence)
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer specializing in creating
        accurate, well-structured, and professional research reports.
        Use ONLY the provided research. Do not fabricate facts, figures, or sources.
        Write in a clear, concise, and objective tone. If data contradicts, note both neutrally."""
    ),
    (
        "human",
        """Write a comprehensive research report on the topic below.

Topic:
{topic}

Research Data:
{research}

Requirements:
1. Begin with a brief Introduction.
2. Present at least 3 well-explained Key Findings with appropriate headings.
3. Include a concise Conclusion summarizing the main insights.
4. List all reference URLs under a Sources section.
5. Ensure the report is factual, coherent, and easy to read.
6. Do not include any information that is not present in the provided research.

Generate the report in Markdown format.
"""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic_chain (refined to hold the writer strictly accountable)
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a ruthless but constructive peer reviewer for scientific and academic text.
        Evaluate reports strictly on:
        - Accuracy (Is everything grounded in the data?)
        - Completeness (Are there 3 clear key findings?)
        - Clarity & Organization
        - Evidence & Sources (Are URLs listed properly?)

        Be constructive, concise, and specific.
        """
    ),
    (
        "human",
        """Review the research report below.

Report:
{report}

Evaluate it based on:
- Accuracy
- Completeness
- Clarity
- Organization
- Evidence & Sources

Respond in the following format:

Overall Score: X/10

Strengths:
- ...

Areas for Improvement:
- ...

Final Verdict:
...
"""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()