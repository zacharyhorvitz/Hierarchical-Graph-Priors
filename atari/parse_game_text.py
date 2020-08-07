from build_graph import *
import sys
import json

with open("game_to_gameplay_text.json","r") as game_file:
    	text_dict = json.load(game_file)

print(text_dict.keys())

print(process_text(text_dict[sys.argv[1]+'\n'],sys.argv[1]))
