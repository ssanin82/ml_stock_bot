"""
Test utterances
----------------
* hi
* show price GOOGL
* compare prices GOOGL AMZN
* list stocks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings
import jinja2
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

from itertools import permutations

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.server import RasaCoreServer

logger = logging.getLogger(__name__)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set()


def new_image_name():
    return 'plot_%d.png' % int(time.time() * 1000)


class ActionShowPrice(Action):
    def name(self):
        return 'action_show_price'

    def resets_topic(self):
        return True

    def run(self, dispatcher, tracker, domain):
        sym = tracker.slots['symbol'].value
        sym = tracker.slots['symbol_compare'].value if not sym else sym
        if sym:
            sym = sym.upper()
            df = load_price_data()
            if sym in df.symbol.unique():
                img = generate_price_plot(sym)
                dispatcher.utter_response({"text": "Here you go!", "image": img})
                plt.show()  # TODO remove later
            else:
                dispatcher.utter_message("No price for symbol %s" % sym)
        else:
            dispatcher.utter_message("Symbol not recognized. Please try again")
        return []


class ActionShowCompare(Action):
    def name(self):
        return 'action_show_compare'

    def resets_topic(self):
        return True

    def run(self, dispatcher, tracker, domain):
        sym1, sym2 = tracker.slots['symbol'].value, tracker.slots['symbol_compare'].value
        if not sym1 or not sym2:
            dispatcher.utter_message("Can't compare")
            if not sym1:
                dispatcher.utter_message("First symbol not found")
            if not sym2:
                dispatcher.utter_message("Second symbol not found")
        else:
            sym1, sym2 = sym1.upper(), sym2.upper()
            df = load_price_data()
            syms = df.symbol.unique()
            if sym1 not in syms or sym2 not in syms:
                dispatcher.utter_message("I have failed to recognize necessary symbols")
                if sym1 not in syms:
                    dispatcher.utter_message("I have no information about symbol %s" % sym1)
                if not sym2:
                    dispatcher.utter_message("I have no information about symbol %s" % sym2)
            else:
                img = generate_compare_plot([sym1, sym2])
                plt.show()  # TODO remove later
                dispatcher.utter_response({"text": "Done!", "image": img})
        return []


class ActionListStocks(Action):
    def name(self):
        return 'action_list_stocks'

    def resets_topic(self):
        return True

    def run(self, dispatcher, tracker, domain):
        df = load_price_data()
        ll = list(df.symbol.unique())
        assert ll, "No stock information was loaded"
        if len(ll) > 1:
            dispatcher.utter_message("I have information about following stocks: %s and %s" %
                                     (', '.join(ll[:-1]), ll[-1]))
        else:
            dispatcher.utter_message("I only know about stock %s" % ll[0])
        return []


class TrainPolicy(KerasPolicy):
    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model


def train_dialogue(domain_file="domain.yml",
                   model_path="models/dialogue",
                   training_data_file="models/stories.md"):
    agent = Agent(domain_file, policies=[MemoizationPolicy(), TrainPolicy()])
    agent.train(training_data_file,
                max_history=1,
                epochs=500,
                batch_size=50,
                augmentation_factor=50,
                validation_split=0.2)
    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data('models/data.md')
    trainer = Trainer(RasaNLUConfig("nlu_model_config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('models/nlu/', fixed_model_name="current")
    return model_directory


def message_preprocessor(msg):
    logger.debug('Preprocessing message: %s' % msg)
    msg = msg.lower()
    logger.debug('Preprocessed message: %s' % msg)
    return msg


def run(serve_forever=True):
    agent = Agent.load("models/dialogue", interpreter=RasaNLUInterpreter("models/nlu/default/current"))
    if serve_forever:
        agent.handle_channel(ConsoleInputChannel(), message_preprocessor)
    return agent


def _permute_chunks(chunks):
    assert len(chunks) == 3
    return [' '.join([c1, c2, c3]).replace('  ', ' ').strip().lower()
            for p in permutations([0, 1, 2])
            for c3 in chunks[p[2]] for c2 in chunks[p[1]] for c1 in chunks[p[0]]]


def _generate_list_stocks():
    articles = ["", "the"]
    syms_synonyms = ["instruments", "stocks", "symbols", "tickers", "what you got", "what you have", "inventory",
                     "items"]
    action = ["list", "show me", "i want to see", "i want you to show me", "show me", "please list", "what are",
              "what are the", "can i see", "how about"]
    return _permute_chunks([action, articles, syms_synonyms])


def _generate_show_stocks(df):
    articles = ["", "the", "a"]
    chunks = list()
    chunks.append(["tell me", "tell me about", "show me", "can i see", "what is", "what's", "show"])
    chunks.append([""] + [("%s price" % a).strip() for a in articles])
    chunks.append([("%s %s %s [%s](symbol)" % (prep, a, w, s)).strip()
                   for prep in ["", "for"]
                   for a in articles
                   for w in ["stock", "symbol", "instrument", "ticker", ""]
                   for s in df.symbol.unique()])
    return _permute_chunks(chunks)


def _generate_cmp_stocks(df):
    chunks = list()
    chunks.append(["compare"])
    chunks.append(["instruments", "stocks", "symbols", "tickers"])
    chunks.append([("[%s](symbol) %s [%s](symbol_compare)" % (s1, conj, s2)).strip()
                   for s1 in df.symbol.unique()
                   for conj in ["", "and", "with", "vs", "versus", "over"]
                   for s2 in df.symbol.unique()])
    return _permute_chunks(chunks)


def generate_data(df):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
    template = env.get_template('template.md')
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    open(os.path.join(models_dir, 'data.md'), 'w').write(template.render(
        list_all=_generate_list_stocks(),
        list_show_stocks=_generate_show_stocks(df),
        list_compare_stocks=_generate_cmp_stocks(df)))


def load_price_data(syms=None):
    # items = ['symbol'] + (['open', 'close', 'low', 'high'] if not items else items)
    items = ['symbol', 'open']  # XXX only 1 price for now
    syms = ["AMZN", "BK", "CSCO", "DOW", "EBAY", "FOX",
            "GOOGL", "HP", "IBM", "JPM", "KMB", "NDAQ"] if not syms else syms
    assert(isinstance(syms, list) or isinstance(syms, set) or isinstance(syms, tuple))
    df = pd.read_csv('./stock_data/prices.csv')
    df = df.loc[df['symbol'].isin(syms)].filter(items=(['date'] + items)).head(100)  # XXX 100 last values
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def generate_price_plot(sym):
    df = load_price_data([sym])
    plt.style.use('seaborn-dark-palette')
    ax = df.plot()
    ax.legend([sym])
    img = new_image_name()
    plt.savefig(img, bbox_inches='tight')
    return img


def generate_compare_plot(syms):
    df = load_price_data(syms)
    plt.style.use('seaborn-dark-palette')
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['symbol']):
        ax.plot(grp.index, grp['open'], label=key)
    ax.legend(syms)
    img = new_image_name()
    plt.savefig(img, bbox_inches='tight')
    return img


def generate_stories():
    blocks = [["* _greet", "   - utter_greet", "   - action_ask_howcanhelp"],
              ["* _show_price", "   - action_show_price", "   - utter_restart"],
              ["* _show_compare", "   - action_show_compare", "   - utter_restart"],
              ["* _list_stocks", "   - action_list_stocks", "   - utter_restart"]]
    bin_arr = []
    bin_str = [0] * len(blocks)
    for i in range(0, 2 ** len(blocks)):
        bin_arr.append("".join(map(str, bin_str))[::-1])
        bin_str[0] += 1
        # Iterate through entire array if there carrying
        for j in range(0, len(bin_str) - 1):
            if bin_str[j] == 2:
                bin_str[j] = 0
                bin_str[j + 1] += 1
                continue
            else:
                break
    ixs = []
    for b in reversed(bin_arr[1:]):
        t = []
        for i in range(len(blocks)):
            if '1' == b[i]:
                t.append(i)
        ixs.append(t)
    ixs_permuted = []
    for x in ixs:
        if len(x) < 2:
            ixs_permuted.append(x)
        else:
            ixs_permuted.extend(permutations(x))
    counter = 0
    lines = []
    for x in ixs_permuted:
        counter += 1
        lines.append("## story_%s" % counter)
        for blk in x:
            lines.extend(blocks[blk])
        lines.append("")
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    open(os.path.join(models_dir, 'stories.md'), 'w').write('\n'.join(lines))


class BotServer(RasaCoreServer):
    def __init__(self, model_directory, nlu_model, verbose, log_file):
        super(BotServer, self).__init__(model_directory, nlu_model, verbose, log_file)


if __name__ == '__main__':
    # logging.basicConfig(level="DEBUG")
    logging.basicConfig(level="INFO")
    parser = argparse.ArgumentParser(description='starts the bot')
    parser.add_argument('task', help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if "train-nlu" == task:
        train_nlu()
    elif "train-dialogue" == task:
        train_dialogue()
    elif "run" == task:
        run()
    elif "generate-data" == task:
        generate_data(load_price_data())
    elif "generate-stories" == task:
        generate_stories()
    elif "server" == task:
        port = 8888
        rasa = BotServer('./models/dialogue', './models/nlu/default/current', True, "./bot.log")
        logger.info("Started http server on port %s" % port)
        rasa.app.run("0.0.0.0", port)
    else:
        warnings.warn("Wrong command line argument")
        exit(1)
