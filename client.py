import requests
import yaml
import random
import sys
# from pprint import pprint
from utils import *


domain = yaml.load(open('domain.yml').read())
# pprint(domain)


def print_action(action):
    reps = domain['templates'][action]
    print(reps[random.randint(0, len(reps) - 1)])


print('Listening...')
while True:
    query = input()
    r = requests.post('http://localhost:8888/conversations/default/parse', json={"query": query})
    # pprint(r.json())

    jsn = r.json()

    if jsn['tracker']['latest_message']['intent']['confidence'] < 0.9:
        print_action('utter_default')
    else:
        intent_name = jsn['tracker']['latest_message']['intent']['name']
        if 'greet' == intent_name:
            print_action('utter_greet')
            print_action('action_ask_howcanhelp')
        elif 'show_price' == intent_name:
            sym = jsn['tracker']['slots']['symbol']
            sym = jsn['tracker']['slots']['symbol_compare'] if not sym else sym
            if sym:
                sym = sym.upper()
                df = load_price_data()
                if sym in df.symbol.unique():
                    generate_price_plot(sym)
                    print("Here you go!")
                    plt.show()  # TODO remove later
                else:
                    print("No price for symbol %s" % sym)
            else:
                print("Symbol not recognized. Please try again")
            print_action('action_ask_howcanhelp')
        elif 'show_compare' == intent_name:
            sym1, sym2 = jsn['tracker']['slots']['symbol'], jsn['tracker']['slots']['symbol_compare']
            if not sym1 or not sym2:
                print("Can't compare")
                if not sym1:
                    print("First symbol not found")
                if not sym2:
                    print("Second symbol not found")
            else:
                sym1, sym2 = sym1.upper(), sym2.upper()
                df = load_price_data()
                syms = df.symbol.unique()
                if sym1 not in syms or sym2 not in syms:
                    print("I have failed to recognize necessary symbols")
                    if sym1 not in syms:
                        print("I have no information about symbol %s" % sym1)
                    if not sym2:
                        print("I have no information about symbol %s" % sym2)
                else:
                    generate_compare_plot([sym1, sym2])
                    print("Done!")
                    plt.show()  # TODO remove later
            print_action('action_ask_howcanhelp')
        elif 'list_stocks' == intent_name:
            df = load_price_data()
            ll = list(df.symbol.unique())
            assert ll, "No stock information was loaded"
            if len(ll) > 1:
                print("I have information about following stocks: %s and %s" % (', '.join(ll[:-1]), ll[-1]))
            else:
                print("I only know about stock %s" % ll[0])
            print_action('action_ask_howcanhelp')
        elif 'exit' == intent_name:
            print_action('utter_goodbye')
            sys.exit(0)
