#%% packages
# install additional packages yfinance, anthropic, panel
import yfinance as yf
import anthropic
import panel as pn
import datetime
pn.extension()
import os
client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API"))

# %% CONSTANTS
MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 1024

# %% user function
def get_stock_price(ticker:str, start='2024-01-01', end='2024-04-26'):
    stock = yf.download(ticker, start=start, end=end)
    return stock

#%% llm function
def get_stock_ticker(description:str):
    response = client.beta.tools.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[
            {
                "name": "get_stock_ticker",
                "description": "Provide the stock ticker for the most probable company which is described in the input text. If in doubt which company to choose, use the company with the highest market capitalization.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker symbol of the company, e.g. TSLA",
                        }
                    },
                    "required": ["ticker"],
                },
            }
        ],
        messages=[{"role": "user", "content": description}],
    )
    return response.content[0].input['ticker']

#%% Get stock performance in this year
def get_stock_performance(question, user, interface):
    current_day = str(datetime.date.today())
    ticker = get_stock_ticker(question)
    df = get_stock_price(ticker, start='2024-01-01', end=current_day)
    price_at_beginning_of_year = df["Close"].iloc[0]
    price_recent = df["Close"].iloc[-1]
    performance_since_beginning_of_year = (price_recent / price_at_beginning_of_year - 1) * 100
    message = client.messages.create(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    messages=[
        {"role": "user", "content": f"You act as stock analyst. Describe the company and the performance for the stock {ticker}, which is {performance_since_beginning_of_year}% since the beginning of the year. They started the year with {price_at_beginning_of_year} and the price on {current_day} was {price_recent} Round the result to one decimal place."}
    ])
    # show the performance
    return message.content[0].text


#%% chat interface
chat_interface = pn.chat.ChatInterface(
    callback = get_stock_performance,
    callback_user = "LLM"
)

chat_interface.send(
    "Describe the company you want to know the stock performance for.",
    user="LLM",
    respond=False
)
# show the chat interface
chat_interface.show()

#%% Create the Panel app
app = pn.Column(chat_interface, sizing_mode="stretch_width")
# Start the server
app.show()



#%% TEST

