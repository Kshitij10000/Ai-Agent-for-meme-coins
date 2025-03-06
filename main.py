import os
import requests
import google.generativeai as genai
import json
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.schema import OutputParserException
from flask import Flask, render_template, request, session, jsonify

# --- Concrete LLM Implementation for Gemini ---
class GeminiLLM(LLM, BaseModel):
    model: str = Field(..., description="The model name for Gemini LLM")
    api_key: str = Field(..., description="API key for accessing Gemini LLM")
    temperature: float = Field(default=0.5)
    verbose: bool = Field(default=False)

    def _call(self, prompt: str, stop=None) -> str:
        genai.configure(api_key=self.api_key)
        model_instance = genai.GenerativeModel(model_name=self.model)
        response = model_instance.generate_content(prompt)
        if self.verbose:
            print("GeminiLLM response:", response.text)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

# --- Provided Web Tools ---
def search_google(query: str):
    try:
        data = json.loads(query)
        if isinstance(data, dict) and "query" in data:
            query = data["query"]
    except json.JSONDecodeError:
        pass
    url = f"https://google.serper.dev/search?q={query}&num=10&apiKey={os.getenv('SERPER_API_KEY')}"
    response = requests.get(url)
    return response.text

def extract_tweet_data(output_json):
    tweets_data = []
    try:
        if isinstance(output_json, str):
            data = json.loads(output_json)
        else:
            data = output_json
        instructions = data.get('result', {}).get('timeline', {}).get('instructions', [])
        for instruction in instructions:
            entries = instruction.get('entries', [])
            for entry in entries:
                if entry.get('content', {}).get('__typename') == 'TimelineTimelineItem':
                    item_content = entry.get('content', {}).get('itemContent', {})
                    if item_content.get('__typename') == 'TimelineTweet':
                        tweet_result = item_content.get('tweet_results', {}).get('result', {})
                        if tweet_result and tweet_result.get('__typename') == 'Tweet':
                            legacy = tweet_result.get('legacy', {})
                            core = tweet_result.get('core', {})
                            user_results = core.get("user_results", {}).get("result", {})
                            tweet_info = {
                                'tweet_id': legacy.get('id_str'),
                                'created_at': legacy.get('created_at'),
                                'full_text': legacy.get('full_text'),
                                'favorite_count': legacy.get('favorite_count'),
                                'retweet_count': legacy.get('retweet_count'),
                                'bookmark_count': legacy.get('bookmark_count'),
                                'language': legacy.get('lang'),
                                'user_id': legacy.get('user_id_str'),
                                'user_screen_name': user_results.get("screen_name"),
                                'is_quote_status': legacy.get("is_quote_status"),
                                'views': tweet_result.get("views", {}).get("count"),
                                'media_urls': [],
                                'hashtags': [],
                                'urls': []
                            }
                            for url_entity in legacy.get('entities', {}).get('urls', []):
                                tweet_info['urls'].append(url_entity.get('expanded_url'))
                            for hashtag_entity in legacy.get('entities', {}).get('hashtags', []):
                                tweet_info['hashtags'].append(hashtag_entity.get('text'))
                            for media_entity in legacy.get('extended_entities', {}).get('media', []):
                                tweet_info['media_urls'].append(media_entity.get('media_url_https'))
                            tweets_data.append(tweet_info)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        print(f"Error processing JSON: {e}")
        return []
    return tweets_data

def fetch_tweets(query: str):
    try:
        data = json.loads(query)
        if isinstance(data, dict) and "query" in data:
            query = data["query"]
    except json.JSONDecodeError:
        pass
    url = "https://twitter241.p.rapidapi.com/search-v2"
    querystring = {"type": "Top", "count": "10", "query": query}
    headers = {
        "x-rapidapi-key": os.getenv("TWITTER_RAPIDAPI_KEY"),
        "x-rapidapi-host": "twitter241.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    clean_tweets = extract_tweet_data(response.json())
    return clean_tweets

def get_coin_data(coin_id):
    try:
        data = json.loads(coin_id)  # Attempt to parse coin_id as JSON
        if isinstance(data, dict) and "coin_id" in data:
            coin_id = data["coin_id"]  # Extract coin_id from JSON
            if not isinstance(coin_id, str):
                raise TypeError("The coin_id extracted from JSON must be a string.")  # Explicit check
        else:
            # If JSON is valid but doesn't contain the expected structure, treat the original as invalid
            raise ValueError("Invalid JSON format.  Expected {'coin_id': 'value'}.")
    except json.JSONDecodeError:
        # If not valid JSON, assume it's already a string
        pass # Keep original coin_id

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Extracting relevant information
        name = data['name']
        symbol = data['symbol']
        description = data['description'].get('en', 'No English description available.')
        homepage = data['links']['homepage'][0] if data['links']['homepage'] else None
        market_cap_rank = data['market_cap_rank']
        current_price_usd = data['market_data']['current_price']['usd']
        ath_usd = data['market_data']['ath']['usd']
        atl_usd = data['market_data']['atl']['usd']
        ath_date = data['market_data']['ath_date']['usd']
        atl_date = data['market_data']['atl_date']['usd']
        price_change_percentage_24h = data['market_data']['price_change_percentage_24h']
        price_change_percentage_7d = data['market_data']['price_change_percentage_7d']
        price_change_percentage_30d = data['market_data']['price_change_percentage_30d']
        price_change_percentage_1y = data['market_data']['price_change_percentage_1y']
        high_24h = data['market_data']['high_24h']['usd']
        low_24h = data['market_data']['low_24h']['usd']
        total_volume = data['market_data']['total_volume']['usd']
        circulating_supply = data['market_data']['circulating_supply']
        max_supply = data['market_data']['max_supply']
        total_supply = data['market_data']['total_supply']

        top_exchanges = []
        if 'tickers' in data and data['tickers']:
            for ticker in data['tickers'][:5]:
                top_exchanges.append({
                    'exchange': ticker['market']['name'],
                    'pair': f"{ticker['base']}/{ticker['target']}",
                    'last_price': ticker['converted_last'].get('usd', 'N/A'),
                    'volume': ticker['volume']
                })
        else:
            top_exchanges = "No ticker information available."

        coin_info = {
            'name': name,
            'symbol': symbol,
            'description': description,
            'homepage': homepage,
            'market_cap_rank': market_cap_rank,
            'current_price_usd': current_price_usd,
            'ath_usd': ath_usd,
            'atl_usd': atl_usd,
            'ath_date': ath_date,
            'atl_date': atl_date,
            'price_change_percentage_24h': price_change_percentage_24h,
            'price_change_percentage_7d': price_change_percentage_7d,
            'price_change_percentage_30d': price_change_percentage_30d,
            'price_change_percentage_1y': price_change_percentage_1y,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'total_volume': total_volume,
            'circulating_supply': circulating_supply,
            'total_supply': total_supply,
            'max_supply': max_supply,
            'top_exchanges': top_exchanges
        }
        return coin_info

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except KeyError as e:
        print(f"Error accessing key: {e}. The API may have changed its structure.")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def check_fraud_coin(coin: str) -> str:
    try:
        data = json.loads(coin)
        if isinstance(data, dict) and "coin" in data:
            coin = data["coin"]
    except json.JSONDecodeError:
        pass
    try:
        with open("fraud_coins.txt", "r") as f:
            frauds = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        return "Error: fraud_coins.txt file not found."
    if coin.lower() in frauds:
        return f"Warning: {coin} is marked as fraudulent."
    else:
        return f"{coin} is not fraudulent."

def llm_sentiment_analysis(tweets: str) -> str:
    prompt = f"""
    You are a financial sentiment analyst. Your task is to first analyze the following tweets step by step, 
    using any available resources if needed, and then provide a concise BUY/SELL/HOLD recommendation.
    Also tell Reason for doing anything and in finale output summarize all the reasoning.

    
    When calling a tool, output exactly in the format below:
    ---
    Action: <Tool Name>
    Action Input: <JSON-formatted input>
    ---
    
    Otherwise, output exactly in the format below:
    ---
    Final Answer: <your answer>
    ---
    
    Ensure that any JSON provided is valid and matches the input requirements of the tool.
    
    Tweets:
    {tweets}
    """
    result = llm(prompt)
    return result.strip()

# --- Wrap Tools for the Agent ---
tools = [
    Tool(
        name="Google Search",
        func=search_google,
        description="Use this tool for generic crypto or coin questions (e.g. 'what is bitcoin?')."
    ),
    Tool(
        name="Fetch Tweets",
        func=fetch_tweets,
        description="Use this tool to fetch recent tweets for a given coin when analyzing sentiment."
    ),
    Tool(
        name="Check Fraud Coin",
        func=check_fraud_coin,
        description="Check if a coin is marked as fraudulent using the fraud_coins.txt file."
    ),
    Tool(
        name="Get Coin Data",
        func=get_coin_data,
        description="Fetch detailed information about a cryptocurrency using the CoinGecko API. "
                    "Provides current price, market cap rank, all-time high/low, and more."
    ),
    Tool(
        name="LLM Sentiment Analysis",
        func=llm_sentiment_analysis,
        description="Analyze tweet sentiments and provide a BUY/SELL/HOLD recommendation using the LLM."
    )
]

# --- Setup Conversation Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Enhanced Agent Prompt Instructions ---
prefix = (
    "You are an advanced crypto recommendation agent. Your goal is to accurately understand and complete any task provided by the user, step by step. "
    "For every request, first analyze and break down the task into clear steps internally, and decide whether you need to use one of the available tools. "
    "When calling a tool, ensure that your input is a valid JSON string exactly as the function expects. "
    "If a user confirmation (e.g., 'yes') is given after a suggestion, use the previous coin reference if available. "
    "If providing a final answer without calling a tool, output exactly in the format:"
    "Also tell Reason for doing anything and in finale output summarize all the reasoning."
    "Final Answer: <your answer>"
    "If calling a tool, output exactly in the format:"
    "Action: <Tool Name>"
    "Action Input: <JSON-formatted input>"
)

suffix = (
    "Conversation History:\n{chat_history}\n"
    "New Question: {input}\n"
    "Process the task step by step using available resources if necessary. {agent_scratchpad}"
)

format_instructions = "Your response MUST be in one of the two formats exactly as specified. Do not include any extra text."

agent_kwargs = {
    "prefix": prefix,
    "suffix": suffix,
    "format_instructions": format_instructions,
}

# --- Initialize the LLM using GeminiLLM ---
llm = GeminiLLM(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    verbose=True
)

# --- Initialize the Agent ---
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    agent_kwargs=agent_kwargs,
    verbose=True,
    memory=memory
)

# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(32)  # Generate a secure secret key

def init_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'current_coin' not in session:
        session['current_coin'] = None

@app.route('/', methods=['GET'])
def index():
    init_chat_history()
    return render_template('chat.html', chat_history=session['chat_history'])

# AJAX endpoint to handle chat messages
@app.route('/send', methods=['POST'])
def send():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data['message'].strip()
    chat_history = session.get('chat_history', [])
    current_coin = session.get('current_coin', None)

    # If the user confirms (e.g., "yes") and no coin is set, try to update context from a previous suggestion.
    if user_message.lower() == "yes" and not current_coin:
        for msg in reversed(chat_history):
            if msg.get("sender") == "Bot" and "did you mean" in msg.get("message", "").lower():
                parts = msg["message"].split("did you mean")
                if len(parts) > 1:
                    coin_candidate = parts[1].split("?")[0].strip().lower()
                    current_coin = coin_candidate
                    session['current_coin'] = current_coin
                    break

    # If the user asks about market cap and a current coin is set, steer the agent to use the Get Coin Data tool.
    if "market cap" in user_message.lower() and current_coin:
        agent_input = f"Get the market cap for {current_coin}."
        response_data = agent.invoke(agent_input)
    else:
        response_data = agent.invoke(user_message)

    # Extract tool info and final answer from the agent response.
    tool_used = None
    final_answer = ""
    if isinstance(response_data, str):
        for line in response_data.splitlines():
            if line.startswith("Action:"):
                tool_used = line.replace("Action:", "").strip()
                break
        final_answer = response_data
    elif isinstance(response_data, dict) and "output" in response_data:
        final_answer = response_data["output"]
    else:
        try:
            final_answer = response_data.content
        except AttributeError:
            final_answer = str(response_data)

    # Update conversation memory.
    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(final_answer)
    chat_history.append({'sender': 'User', 'message': user_message})
    chat_history.append({'sender': 'Bot', 'message': final_answer})
    session['chat_history'] = chat_history

    return jsonify({"user": user_message, "bot": final_answer, "tool": tool_used}), 200

if __name__ == '__main__':
    app.run(debug=True)
