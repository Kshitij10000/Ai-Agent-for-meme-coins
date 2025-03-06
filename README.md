# Memecoin Recommendation Agent

## Overview

This project implements a cryptocurrency recommendation agent that provides BUY/SELL/HOLD recommendations based on real-time data and sentiment analysis. It leverages the Gemini 1.5 Flash LLM, Google Search, Twitter, and a fraud detection database to offer informed insights. The agent is accessible through a Streamlit web application and a Flask-based chat interface.

## Features

-   **Cryptocurrency Analysis:** Provides BUY/SELL/HOLD recommendations for cryptocurrencies.
-   **Sentiment Analysis:** Analyzes Twitter sentiment to gauge public opinion.
-   **Fraud Detection:** Checks coins against a list of known fraudulent coins.
-   **Real-time Data:** Fetches current coin data from the CoinGecko API.
-   **Web Interfaces:**
    -   Streamlit app for a user-friendly interface.
    -   Flask-based chat interface for interactive conversations.

## Technologies Used

-   **LLM:** Gemini 1.5 Flash
-   **LangChain:** For agent implementation and tool integration
-   **Data Sources:**
    -   Google Search (Serper API)
    -   Twitter (Twitter241 RapidAPI)
    -   CoinGecko API
    -   Local fraud\_coins.txt database
-   **Web Frameworks:**
    -   Streamlit
    -   Flask
-   **Python Libraries:**
    -   requests
    -   pydantic
    -   python-dotenv

## Setup Instructions

### Prerequisites

-   Python 3.9+
-   `pip` package installer

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd memecoin_agent
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up API keys:**

    -   Create a `.env` file in the root directory.
    -   Add your API keys to the `.env` file:

        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
        SERPER_API_KEY="YOUR_SERPER_API_KEY"
        TWITTER_RAPIDAPI_KEY="YOUR_TWITTER_RAPIDAPI_KEY"
        RAPIDAPI_KEY="YOUR_RAPIDAPI_KEY"
        ```

    -   Replace `"YOUR_API_KEY"` with your actual API keys.

### Running the Agent

#### Flask App

1.  Navigate to the `bot` directory:

    ```bash
    cd bot
    ```

2.  Run the Flask app.

#### Flask Chat Interface

1.  Navigate to the root directory:

    ```bash
    cd ..
    ```

2.  Run the Flask app:

    ```bash
    python main.py
    ```

3.  Open the Flask app in your browser at `http://127.0.0.1:5000/`.

## Usage

### Flask Chat Interface

1.  Type your message in the input field.
2.  Press "Send" or hit Enter.
3.  The agent's response will appear in the chat window.


## Environment Variables

-   `GOOGLE_API_KEY`: API key for Google Gemini.
-   `TAVILY_API_KEY`: API key for Tavily Search API (if used).
-   `SERPER_API_KEY`: API key for Serper Google Search API.
-   `TWITTER_RAPIDAPI_KEY`: API key for Twitter RapidAPI.
-   `RAPIDAPI_KEY`: Generic RapidAPI key (if needed).

## Fraud Detection Database

The `fraud_coins.txt` file contains a list of coins marked as fraudulent.  Each coin name should be on a new line. The agent checks user queries against this list to warn about potential scams.

## Dependencies

The project dependencies are listed in `requirements.txt`.  Key dependencies include:

-   `langchain`: For building the agent.
-   `google-generativeai`: For interacting with the Gemini LLM.
-   `flask`: For the Flask chat interface.
-   `requests`: For making HTTP requests to external APIs.
-   `python-dotenv`: For loading environment variables from a `.env` file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

[MIT License](LICENSE) (Replace with your chosen license)
