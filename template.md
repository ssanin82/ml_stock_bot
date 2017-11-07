## intent:greet
- hey
- howdy
- hey there
- hi there
- hello there
- hello
- hi
- good morning
- good evening
- good day
- dear sir

## intent:exit
- bye
- exit
- quit
- good bye
- goodbye
- bye bye
- so long
- break
- stop

## intent:list_stocks
{% for s in list_all %}- {{s}}
{% endfor %}

## intent:show_price
{% for s in list_show_stocks %}- {{s}}
{% endfor %}

## intent:show_compare
{% for s in list_compare_stocks %}- {{s}}
{% endfor %}
