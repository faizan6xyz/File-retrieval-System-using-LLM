from duckduckgo_search import DDGS
def search_web(query: str, max_results: int = 5) -> str:
# its a general web search
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(
                f"Title: {r['title']}\n"
                f"URL:   {r['href']}\n"
                f"Info:  {r['body'][:400]}"
            )
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"
def search_news(query: str, max_results: int = 5) -> str:
    """Search recent news only"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        if not results:
            return "No news found."
        output = []
        for r in results:
            output.append(
                f"Title: {r['title']}\n"
                f"URL:   {r['url']}\n"
                f"Date:  {r['date']}\n"
                f"Info:  {r['body'][:400]}"
            )
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"News search error: {e}"
def search_deep(query: str) -> str:
    """
    Searches and returns more results with longer content
    Use this for important sub-topics that need thorough coverage
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(
                f"Title: {r['title']}\n"
                f"URL:   {r['href']}\n"
                f"Info:  {r['body'][:600]}"   # more content per result
            )
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Deep search error: {e}"