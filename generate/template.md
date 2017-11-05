## intent:affirm
- yes
- yep
- yeah
- indeed
- that's right
- ok
- great
- right, thank you
- correct
- great choice
- sounds really good

## intent:deny
- no
- nope
- nah
- no way
- wrong
- incorrect

## intent:goodbye
- bye
- goodbye
- good bye
- stop
- end
- farewell
- Bye bye
- have a good one

## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir

## intent:list_stocks
{% for s in list_all %}- {{s}}
{% endfor %}

## intent:show_price
{% for s in list_show_stocks %}- {{s}}
{% endfor %}

## intent: show_compare
- chart data for symbol [xyz](symbol)
- chart [last year] data for symbol [xyz](symbol)