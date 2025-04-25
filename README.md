# Deep Research System

The **Deep Research System** is an intelligent multi-agent research pipeline powered by LLMs and external search APIs. It is capable of taking a user query and autonomously performing in-depth research using agents for scouting, analysis, writing, and review.

## 🚀 Features

- 🔍 **ScoutAgent**: Generates sub-queries and harvests web data using Tavily search.
- 🧠 **AnalystAgent**: Extracts and synthesizes information from raw web content.
- ✍️ **ScribeAgent**: Drafts coherent and well-structured answers using markdown formatting.
- ✅ **ReviewAgent**: Reviews the answer, checks for factual accuracy, and suggests improvements.
- 🔁 Iterative process with automatic reruns if the review score is below a defined threshold.

## 📦 Requirements

- Python 3.8+
- LangGraph
- LangChain
- OpenAI SDK
- BeautifulSoup
- Requests
- Tavily API Key
- OpenAI API Key
- `python-dotenv`

Install dependencies:

```bash
pip install -r requirements.txt

## Environment Setup

TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
