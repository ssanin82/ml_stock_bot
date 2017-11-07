import requests
import yaml
import random
import sys
from utils import *

domain = yaml.load(open('domain.yml').read())

online = True


def send_out(msg):
    print("Replying: %s" % msg)
    if online:
        requests.patch('http://hedga.herokuapp.com/bot/', json={"result": msg})


def send_image(img):
    print("Sending image file: %s" % img)
    if online:
        requests.post('http://hedga.herokuapp.com/bot/', files={'media': open(img, 'rb')})


def print_action(action):
    reps = domain['templates'][action]
    send_out(reps[random.randint(0, len(reps) - 1)])


send_out('Listening...')
while True:
    if online:
        while True:
            print("polling...")
            r = requests.get("http://hedga.herokuapp.com/bot/dequeue/")
            try:
                jsn = r.json()
            except Exception as e:
                print(e)
                continue
            print(jsn)
            if "sentence" in jsn:
                break
            time.sleep(0.5)
        query = jsn["sentence"]
    else:
        query = input()
    r = requests.post('http://localhost:8888/conversations/default/parse', json={"query": query})

    jsn = r.json()

    if jsn['tracker']['latest_message']['intent']['confidence'] < 0.85:
        print_action('utter_default')
    else:
        intent_name = jsn['tracker']['latest_message']['intent']['name']
        if 'greet' == intent_name:
            print_action('utter_greet')
            print_action('action_ask_howcanhelp')
        elif 'name' == intent_name:
            print_action('utter_name')
            print_action('action_ask_howcanhelp')
        if 'age' == intent_name:
            print_action('utter_age')
            print_action('action_ask_howcanhelp')
        elif 'show_price' == intent_name:
            sym = jsn['tracker']['slots']['symbol']
            sym = jsn['tracker']['slots']['symbol_compare'] if not sym else sym
            if sym:
                sym = sym.upper()
                df = load_price_data()
                if sym in df.symbol.unique():
                    img = generate_price_plot(sym)
                    send_out("Showing price for %s" % sym)
                    if not online:
                        plt.show()
                    else:
                        send_image(img)
                else:
                    send_out("No price for symbol %s" % sym)
            else:
                send_out("No symbols recognized. Please try again")
                print_action('action_ask_howcanhelp')
        elif 'show_compare' == intent_name:
            sym1, sym2 = jsn['tracker']['slots']['symbol'], jsn['tracker']['slots']['symbol_compare']
            if not sym1 or not sym2:
                send_out("Can't compare")
                if not sym1:
                    send_out("First symbol not found")
                if not sym2:
                    send_out("Second symbol not found")
            else:
                sym1, sym2 = sym1.upper(), sym2.upper()
                df = load_price_data()
                syms = df.symbol.unique()
                if sym1 not in syms or sym2 not in syms:
                    send_out("I have failed to recognize necessary symbols")
                    if sym1 not in syms:
                        send_out("I have no information about symbol %s" % sym1)
                    if not sym2:
                        send_out("I have no information about symbol %s" % sym2)
                else:
                    img = generate_compare_plot([sym1, sym2])
                    send_out("Comparing prices for %s and %s" % (sym1, sym2))
                    if not online:
                        plt.show()
                    else:
                        send_image(img)
            print_action('action_ask_howcanhelp')
        elif 'list_stocks' == intent_name:
            df = load_price_data()
            ll = list(df.symbol.unique())
            assert ll, "No stock information was loaded"
            if len(ll) > 1:
                send_out("I have information about following stocks: %s and %s" % (', '.join(ll[:-1]), ll[-1]))
            else:
                send_out("I only know about stock %s" % ll[0])
            print_action('action_ask_howcanhelp')
        elif 'exit' == intent_name:
            print_action('utter_goodbye')
            sys.exit(0)
