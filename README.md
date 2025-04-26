# 📰 AI News Hub

**AI News Hub** is a cutting-edge platform that utilizes artificial intelligence to fetch, summarize, and translate news articles across a wide range of categories.  
Powered by **GPT-4**, it delivers concise and insightful summaries in multiple languages, helping users stay effortlessly informed about global events.

---

## ✨ Features

- 📥 **Fetches** the latest news articles across different categories using **NewsAPI**  
- 🧠 **Generates** high-quality article summaries with **OpenAI’s GPT-4**  
- 🌍 **Supports** multilingual summaries and translations  
- ⚡ **Uses** asynchronous processing for enhanced performance  
- 💾 **Integrates** a caching system for faster and more efficient data access  
- 🖥️ **Provides** a clean, user-friendly interface powered by **Gradio**

---

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ai-news-hub.git
   cd ai-news-hub

2. **Create and activate a virtual environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    
3. **Install the required dependencies**

    ```bash
    pip install -r requirements.txt

4. **Update your API keys**
Replace the placeholder variables **NEWSAPI_KEY**, **OPENAI_API_KEY** with your actual API keys inside the project.

5. **Run the application**

    ```bash
    python3 ai_news_hub.py

6. **Access your application**
Open your browser and navigate to:
    ```bash
    http://127.0.0.1:7860/