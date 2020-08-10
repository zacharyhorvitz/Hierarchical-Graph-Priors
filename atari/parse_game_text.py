from build_graph import *
import sys
import json

with open("game_to_gameplay_text.json","r",encoding="latin-1") as game_file:
    	text_dict = json.load(game_file)

print(text_dict.keys())

if sys.argv[1] == "all":
	print(process_text(" ".join([v.encode("ascii", "ignore").decode() for _,v in text_dict.items()]),sys.argv[1]))


else:
   print(process_text(text_dict[sys.argv[1]+'\n'],sys.argv[1]))
