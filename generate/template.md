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
{% for s in list_compare_stocks %}- {{s}}
{% endfor %}
