#!/usr/bin/env python
"""
RAG Pipeline Module
Pure Retrieval-Augmented Generation (RAG) pipeline for chronic condition coaching.
No external LLM dependency – generation is handled locally using retrieved context.
"""

import logging
from typing import List, Dict, TypedDict
from src.retriever import HealthRetriever

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Type-safe schemas 
class RetrievedContext(TypedDict):
    title: str
    text: str
    source: str
    similarity_score: float


class RAGResult(TypedDict):
    query: str
    recommendations: str
    sources_used: List[Dict[str, float | str]]
    context_count: int


# RAG Pipeline
class RAGPipeline:
    def __init__(self, retriever: HealthRetriever | None = None) -> None:
        """Initialize RAG pipeline with a retriever."""
        self.retriever = retriever or HealthRetriever()
        logger.info("RAGPipeline initialized")

    
    # Retrieval
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedContext]:
        """Retrieve relevant health contexts from vector database."""
        logger.info("Retrieving context for query: %s", query)
        contexts = self.retriever.retrieve_context(query, k=k)
        logger.info("Retrieved %d contexts", len(contexts))
        return contexts

    # Context formatting for debugging 
    def format_context(self, contexts: List[RetrievedContext]) -> str:
        """Format retrieved contexts into a readable string."""
        formatted = "## Relevant Health Knowledge:\n\n"

        for i, ctx in enumerate(contexts, 1):
            formatted += (
                f"{i}. **{ctx['title']}** ({ctx['source']})\n"
                f"   {ctx['text']}\n"
                f"   *Relevance: {ctx['similarity_score']:.2f}*\n\n"
            )

        return formatted


    # Local generation (LLM-free)
    def local_generate_response(
        self,
        user_query: str,
        contexts: List[RetrievedContext]
    ) -> str:
        """
        Generate a response purely from retrieved knowledge.
        This replaces any mock or external LLM dependency.
        """
        logger.info("Generating response locally")

        if not contexts:
            logger.warning("No contexts found for query")
            return (
                "I couldn’t find relevant health knowledge for your question yet. "
                "Try asking about sleep, stress, diet, physical activity, or daily routines."
            )

        response = (
            f"Based on your question about **{user_query}**, here are "
            "evidence-informed lifestyle suggestions:\n\n"
        )

        for i, ctx in enumerate(contexts[:3], start=1):
            response += (
                f"{i}. **{ctx['title']}**\n"
                f"   - Why it matters: {ctx['text'].strip()}\n\n"
            )

        response += (
            "**When to seek professional help:**\n"
            "- If symptoms persist or worsen\n"
            "- Before making major lifestyle changes\n\n"
            " *These recommendations are educational and not a medical diagnosis.*"
        )

        return response


    # End-to-end RAG flow
    def generate_recommendation(
        self,
        user_query: str,
        user_context: str = "",
        k: int = 5
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve → generate.
        """
        logger.info("Starting RAG pipeline for query")

        contexts = self.retrieve(user_query, k=k)

        response = self.local_generate_response(user_query, contexts)

        result: RAGResult = {
            "query": user_query,
            "recommendations": response,
            "sources_used": [
                {
                    "title": ctx["title"],
                    "source": ctx["source"],
                    "relevance": ctx["similarity_score"],
                }
                for ctx in contexts
            ],
            "context_count": len(contexts),
        }

        logger.info("RAG pipeline completed")
        return result



# Simple test harness
if __name__ == "__main__":
    pipeline = RAGPipeline()

    queries = [
        "I have type 2 diabetes and struggle with sleep",
        "How can stress affect high blood pressure?",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        result = pipeline.generate_recommendation(q)
        print(result["recommendations"])