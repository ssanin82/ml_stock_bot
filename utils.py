import time
import jinja2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import permutations

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set()


def new_image_name():
    return 'plot_%d.png' % int(time.time() * 1000)


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
    stories = []

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
    for x in ixs_permuted:
        lines = []
        for blk in x:
            lines.extend(blocks[blk])
        stories.append(lines)

    block = ["* exit", "   - utter_goodbye", "   - action_exit"]
    for i in range(len(stories)):
        story = stories[i][:]
        story.extend(block)
        stories.append(story)

    lines = []
    counter = 0
    for story in stories:
        counter += 1
        lines.append("## story_%s" % counter)
        lines.extend(story)
        lines.append("")
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    open(os.path.join(models_dir, 'stories.md'), 'w').write('\n'.join(lines))
