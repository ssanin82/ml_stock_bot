import requests
import yaml
import random
import sys
from utils import *

domain = yaml.load(open('domain.yml').read())

online = True

"""
#requests.patch('http://hedga.herokuapp.com/bot/api/requests/54/', json={"result": ['123', '456']})
#requests.patch('http://hedga.herokuapp.com/bot/api/requests/54/', json={"result": '456'})
requests.request('PATCH', 'http://hedga.herokuapp.com/bot/api/requests/3/',
                 files = {'file': open('static/img.png', 'rb')},
                 data={"result": 'answer12346'})
sys.exit(1)
"""


def send_out(_id, msgs, img=None):
    print("Replying: %s" % msgs)
    print('With image: %s' % img) if img else None
    if online:
        url = 'http://hedga.herokuapp.com/bot/api/requests/%d/' % _id
        if img:
            requests.request('PATCH', url, files={'file': open(img, 'rb')}, data={"result": ' '.join(msgs)})
        else:
            requests.request('PATCH', url, data={"result": ' '.join(msgs)})


def action_to_repl(action):
    reps = domain['templates'][action]
    return reps[random.randint(0, len(reps) - 1)]


print('Listening...')
_id = 0
while _id < 65:
#while True:
    #_id = -1
    _id += 1
    if online:
        while True:
            print("polling...")
            #r = requests.get("http://hedga.herokuapp.com/bot/dequeue")
            r = requests.get("http://hedga.herokuapp.com/bot/api/requests/%d/" % _id)
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
        _id = jsn['id']
    else:
        query = input()
    r = requests.post('http://localhost:8888/conversations/default/parse', json={"query": query})

    jsn = r.json()

    if jsn['tracker']['latest_message']['intent']['confidence'] < 0.85:
        send_out(_id, [action_to_repl('utter_default')])
    else:
        intent_name = jsn['tracker']['latest_message']['intent']['name']
        if 'greet' == intent_name:
            send_out(_id, [action_to_repl('utter_greet'), action_to_repl('action_ask_howcanhelp')])
        elif 'name' == intent_name:
            send_out(_id, [action_to_repl('utter_name'), action_to_repl('action_ask_howcanhelp')])
        if 'age' == intent_name:
            send_out(_id, [action_to_repl('utter_age'), action_to_repl('action_ask_howcanhelp')])
        elif 'show_price' == intent_name:
            img, lines = None, []
            sym = jsn['tracker']['slots']['symbol']
            sym = jsn['tracker']['slots']['symbol_compare'] if not sym else sym
            if sym:
                sym = sym.upper()
                df = load_price_data()
                if sym in df.symbol.unique():
                    img = generate_price_plot(sym)
                    lines.append("Showing price for %s." % sym)
                    if not online:
                        plt.show()
                else:
                    lines.append("No price for symbol %s." % sym)
            else:
                lines.extend(["No symbols recognized. Please try again.", action_to_repl('action_ask_howcanhelp')])
            send_out(_id, lines, img)
        elif 'show_compare' == intent_name:
            img, lines = None, []
            sym1, sym2 = jsn['tracker']['slots']['symbol'], jsn['tracker']['slots']['symbol_compare']
            if not sym1 or not sym2:
                lines = ["Can't compare."]
                if not sym1:
                    lines.append("First symbol not found.")
                if not sym2:
                    lines.append("Second symbol not found.")
            else:
                sym1, sym2 = sym1.upper(), sym2.upper()
                df = load_price_data()
                syms = df.symbol.unique()
                if sym1 not in syms or sym2 not in syms:
                    lines.append("I have failed to recognize necessary symbols.")
                    if sym1 not in syms:
                        lines.append("I have no information about symbol %s." % sym1)
                    if not sym2:
                        lines.append("I have no information about symbol %s." % sym2)
                else:
                    img = generate_compare_plot([sym1, sym2])
                    lines.append("Comparing prices for %s and %s" % (sym1, sym2))
                    if not online:
                        plt.show()
            lines.append(action_to_repl('action_ask_howcanhelp'))
            send_out(_id, lines, img)
        elif 'list_stocks' == intent_name:
            df = load_price_data()
            ll = list(df.symbol.unique())
            lines = []
            assert ll, "No stock information was loaded."
            if len(ll) > 1:
                lines.append("I have information about following stocks: %s and %s." % (', '.join(ll[:-1]), ll[-1]))
            else:
                lines.append("I only know about stock %s." % ll[0])
            lines.append(action_to_repl('action_ask_howcanhelp'))
            send_out(_id, lines)
        elif 'exit' == intent_name:
            send_out(_id, ['utter_goodbye'])
            sys.exit(0)
