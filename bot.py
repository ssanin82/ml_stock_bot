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

    def run(self, dispatcher, tracker, domain):
        sym = tracker.slots['symbol'].value
        if sym:
            sym = sym.upper()
            df = load_price_data()
            if sym in df.symbol.unique():
                img = generate_price_plot(sym)
                dispatcher.utter_message({"text": "Here you go!", "image": img})
            else:
                dispatcher.utter_message("No price for symbol %s" % sym)
        else:
            dispatcher.utter_message("Symbol not recognized. Please try again")
        return []


class ActionShowCompare(Action):
    def name(self):
        return 'action_show_compare'

    def run(self, dispatcher, tracker, domain):
        sym1, sym2 = tracker.slots['symbol'].value, racker.slots['symbol_compare'].value
        if not sym1 or not sym2:
            dispatcher.utter_message("Can't compare")
            if not sym1:
                dispatcher.utter_message("First symbol not found")
            if not sym2:
                dispatcher.utter_message("Seconf symbol not found")
        else:
            sym1, sym2 = sym1.upper(), sym2.upper()
            df = load_price_data()
            syms = df.symbol.unique()
            if not sym1 in syms or not sym2 in syms:
                dispatcher.utter_message("I failed to recognize necessary symbols")
                if not sym1 in  syms:
                    dispatcher.utter_message("I have no information about symbol %s" % sym1)
                if not sym2:
                    dispatcher.utter_message("I have no information about symbol %s" % sym2)
            else:
                img = generate_compare_plot(sym1, sym2)
                dispatcher.utter_message({"text": "Here you go!", "image": img})
        return []


class ActionListStocks(Action):
    def name(self):
        return 'action_list_stocks'

    def run(self, dispatcher, tracker, domain):
        df = load_price_data()
        dispatcher.utter_message("Stock list: %s" % df.symbol.unique())
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


def train_dialogue(domain_file="domain.yml", model_path="models/dialogue", training_data_file="data/stories.md"):
    agent = Agent(domain_file, policies=[MemoizationPolicy(), TrainPolicy()])
    agent.train(training_data_file,
                max_history=4,
                epochs=300,
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


def run(serve_forever=True):
    agent = Agent.load("models/dialogue",
                       interpreter=RasaNLUInterpreter("models/nlu/default/current"))
    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


def _permute_chunks(chunks):
    assert len(chunks) == 3
    return [' '.join([c1, c2, c3]).replace('  ', ' ').strip()
            for p in permutations([0, 1, 2])
            for c3 in chunks[p[2]] for c2 in chunks[p[1]] for c1 in chunks[p[0]]]


def _generate_list_stocks():
    articles = ["", "the"]
    syms_synonyms = ["instruments", "stocks", "symbols", "tickers"]
    action = ["list", "show me", "i want to see", "i want you to show me", "show me", "please list", "what are",
              "what are the", "can i see", "how about"]
    return _permute_chunks([action, articles, syms_synonyms])


def _generate_show_stocks(df):
    articles = ["", "the", "a"]
    chunks = list()
    chunks.append(["show me", "can i see", "what is", "what's", "show"])
    chunks.append([("%s price" % a).strip() for a in articles])
    chunks.append([("%s %s %s [%s](symbol)" % (prep, a, w, s)).strip()
                   for prep in ["", "for"]
                   for a in articles
                   for w in ["stock", "symbol", "instrument", "ticker"]
                   for s in df.symbol.unique()])
    return _permute_chunks(chunks)


def generate_data(df):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('./generate'))
    template = env.get_template('template.md')
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    open(os.path.join(models_dir, 'data.md'), 'w').write(template.render(
        list_all=_generate_list_stocks(),
        list_show_stocks=_generate_show_stocks(df)))


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


def generate_compare_plot(sym1, sym2):
    df = load_price_data([sym1, sym2])
    plt.style.use('seaborn-dark-palette')
    ax = df.plot()
    ax.legend([sym1, sym2])
    img = new_image_name()
    plt.savefig(img, bbox_inches='tight')
    return img


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    # logging.basicConfig(level="INFO")
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
    else:
        warnings.warn("Wrong command line argument")
        exit(1)
