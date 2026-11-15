from langchain_core.tools import tool
import arxiv

@tool
def search_arxiv(query: str) -> str:
    """Search academic papers on ArXiv. Use this when the user asks for recent research, papers, or topics not found in the local uploaded documents."""
    try:
        search = arxiv.Search(query=query, max_results=3)
        results = [
            f"Title: {doc.title}\nSummary: {doc.summary[:300]}..." 
            for doc in search.results()
        ]
        return "\n\n".join(results) if results else "No relevant papers found on ArXiv."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"