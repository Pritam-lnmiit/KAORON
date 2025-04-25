# File: deep_research_system/main.py

import os
import json
import uuid
import re
import logging
from typing import Dict, List, TypedDict, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from datetime import datetime

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    # Set up logging configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/research_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    return logging.getLogger("deep_research")

logger = setup_logging()

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM and Tavily Search
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Define State for LangGraph
class ResearchState(TypedDict):
    query: str
    sub_queries: List[str]
    raw_data: List[Dict]
    processed_data: List[Dict]
    synthesized_data: Dict
    draft_answer: str
    final_answer: str
    review_score: Optional[float]
    iteration: int

# ScoutAgent: Web Crawler & Data Harvester
def scout_agent(state: ResearchState) -> ResearchState:
    query = state["query"]
    iteration = state.get("iteration", 1)
    logger.info(f"ScoutAgent (Iteration {iteration}): Processing query '{query}'")

    # Break down query into sub-queries
    sub_query_prompt = ChatPromptTemplate.from_template(
        """
        Given the main query: '{query}', generate 3 focused sub-queries to aid in comprehensive research.
        Provide the sub-queries as a JSON list.
        """
    )
    chain = sub_query_prompt | llm
    try:
        sub_queries_raw = chain.invoke({"query": query}).content
        logger.info(f"Raw sub-queries: {sub_queries_raw}")
        # Strip markdown and whitespace
        sub_queries_clean = re.sub(r'```json\s*|\s*```', '', sub_queries_raw).strip()
        sub_queries = json.loads(sub_queries_clean) if isinstance(sub_queries_clean, str) else sub_queries_clean
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}. Falling back to default sub-queries.")
        sub_queries = [
            query,
            f"Recent breakthroughs in {query}",
            f"Challenges and future of {query}"
        ]  # Default sub-queries

    # Perform searches
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    raw_data = []
    for sub_query in sub_queries:
        try:
            search_results = tavily_tool.invoke(sub_query)
            for result in search_results:
                # Fetch full content for deeper analysis
                try:
                    response = requests.get(result["url"], timeout=5, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    full_content = soup.get_text(separator=" ", strip=True)[:5000]  # Limit size
                except (requests.RequestException, ValueError) as e:
                    logger.warning(f"Error fetching {result['url']}: {e}")
                    full_content = result["content"]
                
                raw_data.append({
                    "url": result["url"],
                    "title": result.get("title", "No title"),
                    "content": full_content,
                    "snippet": result.get("snippet", ""),
                    "sub_query": sub_query
                })
        except Exception as e:
            logger.error(f"Error processing sub-query '{sub_query}': {e}")
            continue

    return {
        **state,
        "sub_queries": sub_queries,
        "raw_data": state.get("raw_data", []) + raw_data,
        "iteration": iteration
    }

# AnalystAgent: Information Extractor & Synthesizer
def analyst_agent(state: ResearchState) -> ResearchState:
    raw_data = state["raw_data"]
    query = state["query"]
    logger.info("AnalystAgent: Extracting and synthesizing information")

    # Split and process raw data
    processed_data = []
    for item in raw_data:
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            processed_data.append({
                "url": item["url"],
                "title": item["title"],
                "content": chunk,
                "sub_query": item["sub_query"],
                "metadata": {"timestamp": datetime.now().isoformat()}
            })

    # Summarize and extract key points
    summary_prompt = ChatPromptTemplate.from_template(
        """
        Summarize the following content and extract key points relevant to the query: '{query}'.
        Content: {content}
        
        **Output Format (JSON):**
        {{
            "summary": "Brief summary (100-150 words)",
            "key_points": ["Point 1", "Point 2", ...],
            "entities": ["Entity 1", "Entity 2", ...]
        }}
        """
    )
    chain = summary_prompt | llm
    synthesized_data = {}
    for item in processed_data:
        try:
            result_raw = chain.invoke({"query": query, "content": item["content"]}).content
            logger.info(f"Raw summary for {item['url']}: {result_raw}")
            result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
            synthesized_data[item["url"]] = result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {item['url']}: {e}. Using fallback summary.")
            synthesized_data[item["url"]] = {
                "summary": item["content"][:150] + "..." if len(item["content"]) > 150 else item["content"],
                "key_points": ["Information not fully processed due to parsing error"],
                "entities": []
            }

    return {
        **state,
        "processed_data": processed_data,
        "synthesized_data": synthesized_data
    }

# ScribeAgent: Answer Drafter
def scribe_agent(state: ResearchState) -> ResearchState:
    query = state["query"]
    synthesized_data = state["synthesized_data"]
    logger.info("ScribeAgent: Drafting answer")

    # Prepare context
    context = "\n".join(
        f"Source: {url}\nSummary: {data['summary']}\nKey Points: {', '.join(data['key_points'])}\nEntities: {', '.join(data['entities'])}"
        for url, data in synthesized_data.items()
    )

    # Draft answer
    draft_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert writer. Based on the following synthesized research data, provide a coherent, concise, and accurate response to the query: '{query}'.
        
        **Research Data:**
        {context}
        
        **Instructions:**
        - Write a detailed response (200-300 words).
        - Cite sources by URL.
        - Use markdown formatting.
        - Ensure clarity and logical flow.
        
        **Response:**
        """
    )
    chain = draft_prompt | llm
    draft_answer = chain.invoke({"query": query, "context": context}).content

    return {
        **state,
        "draft_answer": draft_answer,
        "final_answer": draft_answer
    }

# ReviewAgent: Fact Checker & Enhancer
def review_agent(state: ResearchState) -> ResearchState:
    query = state["query"]
    draft_answer = state["draft_answer"]
    raw_data = state["raw_data"]
    logger.info("ReviewAgent: Reviewing draft answer")

    # Evaluate answer quality
    review_prompt = ChatPromptTemplate.from_template(
        """
        Review the following draft answer for the query: '{query}'.
        Draft: {draft}
        
        **Source Data:**
        {source_data}
        
        **Instructions:**
        - Evaluate factual accuracy, coherence, and completeness.
        - Assign a score (0-100) based on quality.
        - Provide feedback and suggest improvements.
        
        **Output (JSON):**
        {{
            "score": float,
            "feedback": "Feedback text",
            "needs_revision": bool
        }}
        """
    )
    source_data = "\n".join(f"Source: {item['url']}\nContent: {item['content'][:500]}" for item in raw_data)
    chain = review_prompt | llm
    try:
        review_result_raw = chain.invoke({"query": query, "draft": draft_answer, "source_data": source_data}).content
        logger.info(f"Raw review result: {review_result_raw}")
        review_result = json.loads(review_result_raw) if isinstance(review_result_raw, str) else review_result_raw
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in review: {e}. Assuming low score.")
        review_result = {"score": 50, "feedback": "Malformed review output.", "needs_revision": True}

    return {
        **state,
        "review_score": review_result["score"],
        "final_answer": draft_answer if not review_result["needs_revision"] else state["final_answer"],
        "iteration": state["iteration"] + 1 if review_result["needs_revision"] else state["iteration"]
    }

# Conditional routing for review
def route_review(state: ResearchState) -> str:
    score = state.get("review_score", 0)
    iteration = state.get("iteration", 1)
    if score < 80 and iteration <= 3:  # Max 3 iterations
        logger.info(f"ReviewAgent: Score {score} < 80, triggering re-research (Iteration {iteration})")
        return "scout"
    return END

# Define the workflow using LangGraph
def create_workflow():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("scout", scout_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("scribe", scribe_agent)
    workflow.add_node("review", review_agent)
    
    # Define edges
    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "analyst")
    workflow.add_edge("analyst", "scribe")
    workflow.add_edge("scribe", "review")
    workflow.add_conditional_edges("review", route_review, {"scout": "scout", END: END})
    
    return workflow.compile()

# Main function to run the system
def run_deep_research(query: str) -> Dict:
    workflow = create_workflow()
    initial_state = {
        "query": query,
        "sub_queries": [],
        "raw_data": [],
        "processed_data": [],
        "synthesized_data": {},
        "draft_answer": "",
        "final_answer": "",
        "review_score": None,
        "iteration": 1
    }
    
    result = workflow.invoke(initial_state)
    
    # Save the full results to a JSON file
    result_id = uuid.uuid4()
    result_file = f"results/research_{result_id}.json"
    
    # Save a simplified version with only essential data (to keep file size manageable)
    simplified_result = {
        "query": result["query"],
        "sub_queries": result["sub_queries"],
        "final_answer": result["final_answer"],
        "review_score": result["review_score"],
        "iteration": result["iteration"],
        "sources": [{"title": item["title"], "url": item["url"]} for item in result["raw_data"]],
        "synthesized_data": result["synthesized_data"]
    }
    
    with open(result_file, "w") as f:
        json.dump(simplified_result, f, indent=2)
        
    logger.info(f"Full results saved to {result_file}")
    
    return result

# Example usage
if __name__ == "__main__":
    query = "What are the latest advancements in super computers"
    result = run_deep_research(query)
    
    logger.info("\n=== Deep Research System Output ===")
    logger.info(f"Query: {result['query']}")
    logger.info(f"Final Answer:\n{result['final_answer']}")
    logger.info(f"Review Score: {result['review_score']}")
    logger.info(f"Results saved to results/research_{uuid.uuid4()}.json")
    
    # Also print a nice summary to the console
    print("\n" + "="*50)
    print("DEEP RESEARCH SYSTEM - RESULTS SUMMARY")
    print("="*50)
    print(f"Query: {result['query']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Review Score: {result['review_score']}")
    print("\nSub-queries:")
    for i, sub_query in enumerate(result['sub_queries'], 1):
        print(f"  {i}. {sub_query}")
    
    print("\nSources:")
    for i, data in enumerate(result["raw_data"], 1):
        print(f"  {i}. {data['title']} ({data['url']})")
    
    print("\nFinal Answer:")
    print("-"*50)
    print(result['final_answer'])
    print("-"*50)
    print(f"\nDetailed results saved to: results/research_{uuid.uuid4()}.json")
    print("\nLog file saved to: logs/")