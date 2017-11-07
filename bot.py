# Example of passing intents to the bot directly: _show_compare[symbol=AMZN,symbol_compare=BK]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from utils import *

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.server import RasaCoreServer

logger = logging.getLogger(__name__)


class ActionExit(Action):
    def name(self):
        return 'action_exit'

    def resets_topic(self):
        return True

    def run(self, dispatcher, tracker, domain):
        exit()
        return []


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

        n_hidden = 300  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        logger.debug(model.summary())
        return model


def train_dialogue(domain_file="domain.yml",
                   model_path="models/dialogue",
                   training_data_file="models/stories.md"):
    agent = Agent(domain_file, policies=[MemoizationPolicy(), TrainPolicy()])
    # agent = Agent(domain_file, policies=[TrainPolicy()])
    agent.train(training_data_file,
                max_history=3,
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


def on_circuit_break(tracker, dispatcher):
    pass  # XXX not sure if we need it for now. It is called when max_number_of_predictions is exceeded


def run(serve_forever=True):
    agent = Agent.load("models/dialogue", interpreter=RasaNLUInterpreter("models/nlu/default/current"))
    if serve_forever:
        agent.handle_channel(ConsoleInputChannel(), message_preprocessor)
    return agent


if __name__ == '__main__':
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
    elif "run-debug" == task:  # TODO separate flag
        logging.basicConfig(level="DEBUG")
        run()
    elif "generate-data" == task:
        generate_data(load_price_data())
    elif "generate-stories" == task:
        generate_stories()
    elif "server" == task:
        port = 8888
        rasa = RasaCoreServer('./models/dialogue', './models/nlu/default/current', True, "./bot.log")
        logger.info("Started http server on port %s" % port)
        rasa.app.run("0.0.0.0", port)
    else:
        warnings.warn("Wrong command line argument")
        exit(1)
