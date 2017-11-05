# ml_stock_bot

Puprose: create a chat bot to hold conversation about stock prices

Currently supports:
* listing stocks
* requesting stock price
* comparing 2 stocks

Restrictions:
1. No temporal expression recognition (only shows last 100 values)
2. Can only show stock open price, no other parameters
3. Can only show 1 stock at a time
4. Can only compare 2 stocks at a time
5. For simplicity dialog history is kept to 1 in bot (remembering previous states requires more effort to train the bot correctly predict next dialog frame)
6. Stock names is a part of training which reduces this bot's applicability in real-life situations

TODO:
1. Need to implement "I don't understand" case, otherwise whatever is told is mapped to one of existing actions
2. Stock names are case-sensitive. Need to find out how to handle this
3. Need to disable all predictions for now - sometimes it may cause infinite loop. Only direct mapping of dialogue states

---
Stock data loaded from [kaggle](https://www.kaggle.com/dgawlik/nyse/data)

