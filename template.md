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

## intent:name
- tell me you name
- what is your name
- your name please
- my name is john, what is your name
- my name is john
- tell me about your name
- would you tell me your name
- please tell me your name
- and your name is
- how about your name
- tell me about your name
- is your name john
- are you not john by name
- is john not your name
- i want to know your name
- i would like to know your name
- what is it about your name
- name please
- please name yourself
- who are you
- tell me who you are
- tell me about your name
- tell me about who you are

## intent:age
- how old are you
- what is your age
- are you old
- tell me about your age
- tell me how old are you
- how about your age
- disclose your age to me
- tell me how old are you
- tell me your age
- tell me are you old
- when were you born
- are you young
- tell me when you were born
- how young are you
- are you young
- are you old or young
- are you young or old
- were you born long ago
- tell me when you were born
- tell me whether you are young or old
