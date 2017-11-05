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
            load_price_data()
            sym = sym.upper()
            if sym in prices.keys():
                pass  # TODO
                dispatcher.utter_message("Here you go!")
            else:
                dispatcher.utter_message("No data for symbol %s" % sym)
        else:
            dispatcher.utter_message("Symbol not recognized. Please try again")
        return []


class ActionShowCompare(Action):
    def name(self):
        return 'action_show_compare'

    def run(self, dispatcher, tracker, domain):
        sym = tracker.slots['symbol'].value
        if sym:
            c = generate_price_data()
            sym = sym.upper()
            if sym in prices.keys():
                pass  # TODO
                dispatcher.utter_response("Here you go!")
            else:
                dispatcher.utter_message("No data for symbol %s" % sym)
        else:
            dispatcher.utter_message("Symbol not recognized. Please try again")
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


def generate_show_volume(df):
    articles = ["", "the", "a"]
    chunks = list()
    chunks.append(["show me", "can i see", "what is", "what's", "show"])
    chunks.append([("%s volume" % a).strip() for a in articles])
    chunks.append([("%s %s %s [%s](symbol)" % (prep, a, w, s)).strip()
                   for prep in ["", "for"]
                   for a in articles
                   for w in ["stock", "symbol", "instrument", "ticker"]
                   for s in df.symbol.unique()])

    lines = list()
    for p in permutations([0, 1, 2]):
        for c1 in chunks[p[0]]:
            for c2 in chunks[p[1]]:
                for c3 in chunks[p[2]]:
                    lines.append(' '.join([c1, c2, c3]).replace('  ', ' '))

    return lines


def generate_data(df):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('./generate'))
    template = env.get_template('template.md')
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    open(os.path.join(models_dir, 'data.md'), 'w').write(template.render(
        list_show_volume=generate_show_volume(df)))


def load_price_data(syms=None):
    # items = ['open', 'close', 'low', 'high'] if not items else items
    items = ['open']  # XXX only 1 price for now
    syms = ["AMZN", "BBBY", "CAG", "DHI", "EBAY", "FITB",
            "GOOGL", "HST", "ILMN", "JPM", "KMB", "LUK"] if not syms else syms
    assert(isinstance(syms, list) or isinstance(syms, set) or isinstance(syms, tuple))
    df = pd.read_csv('./stock_data/prices.csv')
    df = df.loc[df['symbol'].isin(syms)]
    df = df.loc[df['symbol'] in syms].filter(items=(['date'] + items)).head(100)  # XXX 100 last values
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
        pass  #  TODO generate_data(load_price_data())
    else:
        warnings.warn("Wrong command line argument")
        exit(1)
