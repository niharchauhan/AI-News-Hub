import os
import asyncio
import aiohttp
import math
from newsapi import NewsApiClient
import logging
from openai import AsyncOpenAI
import gradio as gr
import traceback
from collections import OrderedDict

NEWSAPI_KEY = "your-key"
OPENAI_API_KEY = "your-key"

# Configure detailed logging to debug some errors
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if any(key is None for key in [NEWSAPI_KEY, OPENAI_API_KEY]):
    logging.error("API keys are missing. Please check your environment variables.")

# Start the AsyncOpenAI client using our OPENAI API key.
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Simple cache implementation
class SimpleCache(OrderedDict):
    def __init__(self, max_size=100):
        super().__init__()
        self.max_size = max_size
        logging.info(f"Cache initialized with max size: {self.max_size}")

    async def get(self, key, create_func):
        if key in self:
            self.move_to_end(key)
            logging.info(f"Cache hit for key: {key}")
        else:
            logging.info(f"Cache miss for key: {key}. Creating new entry...")
            self[key] = await create_func()
            if len(self) > self.max_size:
                removed_key, _ = self.popitem(last=False)
                logging.warning(f"Cache full. Removed oldest entry: {removed_key}")

        return self[key]

# Create a cache instance
summary_cache = SimpleCache(max_size=100)

def fetch_global_headlines(category='general', page_size=15):
    """Retrieve news headlines from NewsAPI."""
    def get_articles(response):
        return response.get('articles', [])

    try:
        top_headlines = NewsApiClient(api_key=NEWSAPI_KEY).get_top_headlines(
            category=category,
            language='en',
            page_size=page_size
        )

        articles = get_articles(top_headlines)
        logging.info(f"Response received from NewsAPI. {len(articles)} articles were retrieved.")
        return articles

    except Exception as e:
        logging.error(f"Error fetching headlines: {e}", exc_info=True)
        return []

async def send_warmup_request():
    """Send a minimal request to OpenAI to warm up the connection."""
    return await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Warm-up request"}],
        max_tokens=5
    )

async def warm_up_openai():
    """Warm up the OpenAI API connection by sending a test request."""
    try:
        await send_warmup_request()
        logging.info("OpenAI connection warmed up successfully.")
    except Exception as e:
        logging.error(f"Failed to warm up OpenAI connection: {e}")
        logging.error(traceback.format_exc())

async def translate_text(text: str, target_language: str) -> str:
    """Translate the given text to the target language using GPT-4."""
    system_prompt = (
        f"You are a professional translator. Translate the following text to {target_language}."
    )
    user_prompt = {"role": "user", "content": text}
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                user_prompt
            ],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error translating text: {str(e)}")
        return text

async def process_article(article, language):
    """Process a single article: extract info, generate summary, and translate title if needed."""
    try:
        # Extract article metadata
        source = article.get('source', {}).get('name', '[Source Unavailable]')
        title = article.get('title', '[Title Unavailable]')
        description = article.get('description') or article.get('content') or title
        url = article.get('url', '#')
        image_url = article.get('urlToImage', '')

        # Check for sufficient content
        if not title or not description or len(description) < 50:
            logging.warning(f"Skipping article due to insufficient content: {title}")
            return generate_html_card(
                title=title,
                url=url,
                source=source,
                summary="Content unavailable for this article.",
                image_url=None
            )

        # Generate a cache key and get or generate the summary
        cache_key = f"{description[:1000]}_{language}"
        summary = await summary_cache.get(cache_key, lambda: summarize_article(description, language))

        if len(summary) < 50:
            summary = f"Summary unavailable. Please read the full article at: {url}"

        # Translate title if necessary
        if language.lower() != "english":
            title = await translate_text(title, language)

        return generate_html_card(
            title=title,
            url=url,
            source=source,
            summary=summary,
            image_url=image_url
        )

    except Exception as e:
        logging.error(f"Error processing article: {str(e)}")
        logging.error(traceback.format_exc())
        return f"<div>Error processing article: {str(e)}</div>"

async def summarize_article(article_text: str, language: str) -> str:
    """Asynchronously summarize an article using OpenAI's GPT-4 model and translate if needed."""
    system_prompt = (
        "You are an expert news analyst and summarizer. Provide concise, insightful summaries that capture "
        "the core of news articles, including key events, figures, and implications."
    )

    user_prompt = (
        f"Summarize this news article in 2-3 sentences. Highlight the main event, key figures, and any "
        f"significant impacts or outcomes. Ensure the summary is informative and contextual. "
        f"Article: {article_text[:1000]}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()

        if language.lower() != "english":
            summary = await translate_text(summary, language)

        return summary

    except Exception as e:
        logging.error(f"Error summarizing article: {str(e)}")
        logging.error(traceback.format_exc())
        return (
            "Unable to generate summary due to an error. "
            "Please refer to the original article for information."
        )

async def news_aggregator_async(category, language):
    """Main asynchronous function to aggregate and summarize news."""
    try:
        await warm_up_openai()

        logging.info(f"Fetching headlines for category: {category}")
        articles = fetch_global_headlines(category=category)
        
        if not articles:
            return "No news articles found for this category. Please try another category."

        logging.info(f"We have got {len(articles)} articles. Now they are processing...")

        summarized_articles = await asyncio.gather(*[process_article(article, language) for article in articles])

        if not summarized_articles:
            return "Error processing news articles. Please try again later."

        logging.info(f"Successfully processed {len(summarized_articles)} articles")
        return "".join(summarized_articles)
    except Exception as e:
        logging.error(f"Error in news_aggregator_async: {str(e)}")
        logging.error(traceback.format_exc())
        return "We are experiencing technical difficulties. Please try again later or choose a different category."

def news_aggregator(category, language):
    if not category or not language:
        return "Please select a news category and language."
    try:
        return asyncio.run(news_aggregator_async(category, language))
    except Exception as e:
        logging.error(f"Error in news_aggregator: {str(e)}")
        logging.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

def generate_html_card(title, url, source, summary, image_url=None):
    """
    Generate an HTML card for the article.
    """
    image_html = f'<img src="{image_url}" alt="Article image" class="article-image">' if image_url else ''
    return f"""
    <div class='article-card'>
        <h3><a href='{url}' target='_blank'>{title}</a></h3>
        <p><strong>Media:</strong> {source}</p>
        <p><strong>Overview:</strong> {summary}</p>
        {image_html}
    </div>
    """

# News Categories list
CATEGORIES = [
    "business", "entertainment", "general", "health", "science", "sports", "technology"
]

# Languages supported
LANGUAGES = ["English", "Spanish", "French", "German", "Chinese", "Hindi", "Arabic"]

# CSS styles for the Gradio interface
css = """
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background-image: url('https://st3.depositphotos.com/29296402/37009/v/450/depositphotos_370098862-stock-illustration-the-news-sketch-vector-seamless.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    font-family: Arial, sans-serif;
}

#gradio-container {
    width: 100%;
    max-width: 1200px;
    padding: 20px;
    box-sizing: border-box;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* Centering the header text */
.center-header {
    text-align: center;
    font-size: 2em;
    font-weight: bold;
    color: #333;
    margin: 0;
    padding: 10px 0;
}

/* Blue button styling */
.blue-button {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 1em;
    border-radius: 5px;
    cursor: pointer;
}

.blue-button:hover {
    background-color: #0056b3;
}

/* Article card styles */
.article-card {
    border: 1px solid #ddd;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 12px;
    background: #f9f9f9;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    overflow: hidden;
    font-family: Arial, sans-serif;
}

.article-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

.article-card h3 {
    margin-top: 0;
    font-size: 20px;
    font-weight: bold;
    color: #007bff;
    margin-bottom: 10px;
    word-break: break-word;
}

.article-card h3 a {
    text-decoration: none;
    color: #007bff;
    transition: color 0.3s ease;
}

.article-card h3 a:hover {
    color: #0056b3;
}

.article-card p {
    margin: 5px 0;
    color: #555;
    line-height: 1.6;
}

.article-card strong {
    color: #333;
    font-weight: bold;
}

.article-image {
    max-width: 100%;
    height: auto;
    margin-top: 10px;
    border-radius: 8px;
    border: 1px solid #ddd;
}

/* Scrollable output container */
#output-container {
    max-height: 600px;
    overflow-y: auto;
    margin-top: 20px;
    padding: 10px;
    border-radius: 8px;
    background-color: rgba(255, 255, 255, 0.9);
}
"""

#Gradio Interface with centered header and blue button
with gr.Blocks(css=css, elem_id="gradio-container") as iface:
    # Moving header outside the nested Block
    gr.HTML("<div class='center-header' style='margin-top: 15px;'>AI News Hub</div>")
    
    # Add a description/Markdown for the app
    gr.Markdown("Get summarized news headlines across various categories in different languages.")
    
    with gr.Row():
        with gr.Column(scale=1):
            category_input = gr.Dropdown(choices=CATEGORIES, label="Pick a News Category")
            language_input = gr.Dropdown(choices=LANGUAGES, label="Choose a Language", value="English")
            submit_button = gr.Button("Show Me the News", elem_classes=["blue-button"])  # Add class for styling
        
        with gr.Column(scale=2):
            output = gr.HTML(elem_id="output-container")
    
    submit_button.click(fn=news_aggregator, inputs=[category_input, language_input], outputs=output)
    
    submit_button.click(fn=lambda: output.update(''), inputs=[], outputs=output)

iface.launch()