# world_sentiment_bot.py

import asyncio
import json
from datetime import datetime
from duckduckgo_search import DDGS
from openai import AsyncOpenAI

# ==== SETUP ====

client = AsyncOpenAI()

countries = ["USA", "India", "Germany", "Brazil", "China"]
categories = ["politics", "economy", "technology", "health", "conflict", "climate change"]

today = datetime.now().strftime('%Y-%m-%d')

# ==== TOOLS ====

async def search_news(topic: str, max_results: int = 5) -> list:
    with DDGS() as ddg:
        results = ddg.text(f"{topic} news {today}", max_results=max_results)
        return results or []

async def summarize_and_analyze(articles: list) -> dict:
    """Summarize and analyze sentiment using OpenAI."""
    if not articles:
        return {"summary": "No news found.", "sentiment": "neutral"}

    text_block = "\n\n".join(
        f"Title: {a['title']}\nSummary: {a['body']}" for a in articles
    )

    prompt = f"""
You are a world news analyst.

Summarize the following news articles in 3-5 bullet points.

Also, determine the **overall sentiment** (positive, neutral, or negative) based on the tone and news contents.

News Articles:
{text_block}

# Your Output:
- Summary (bullet points)
- Overall Sentiment (positive / neutral / negative)
"""
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    output = response.choices[0].message.content
    # Very basic split
    if "positive" in output.lower():
        sentiment = "positive"
    elif "negative" in output.lower():
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"raw": output, "sentiment": sentiment}

# ==== MAIN RUN ====

async def run_sentiment_bot():
    report = {}
    for country in countries:
        report[country] = {}
        for category in categories:
            query = f"{category} news {country}"
            print(f"Searching news for {query}...")
            articles = await search_news(query)
            print(f"Found {len(articles)} articles. Summarizing...")
            summary_sentiment = await summarize_and_analyze(articles)
            report[country][category] = summary_sentiment
            await asyncio.sleep(2)  # small sleep to avoid hammering

    # Save report
    filename = f"world_sentiment_{today}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ World Sentiment Report generated: {filename}")

# ==== SCRIPT ENTRY ====

if __name__ == "__main__":
    asyncio.run(run_sentiment_bot())
