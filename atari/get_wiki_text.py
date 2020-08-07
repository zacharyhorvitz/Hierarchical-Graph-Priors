import urllib.request
import urllib.error
from lxml import etree
import numpy as np
import requests
import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
import json
import random
from tqdm import tqdm
import unicodedata
import re
import wikipedia

def requester(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,} 

    request=urllib.request.Request(url,None,headers) #The assembled request
    response = urllib.request.urlopen(request)
    data = response.read() # The data u need
    soup = BeautifulSoup(data, 'lxml')

    return soup

if __name__ == "__main__":


    with open("atari_titles.txt","r") as atari_file:
        all_games = atari_file.readlines()

    game_to_gameplay = {}

    for g in all_games:
        titles = wikipedia.search(g+" ( Video Game )")
        titles = [t for t in titles if "List of" not in t]
        if len(titles) == 0: continue

        title = titles[0]
        print(title)

        try:
	        page_data = wikipedia.page(title,auto_suggest=False).content
	                
	        if "== Gameplay ==" in page_data:
	            location = page_data.index("== Gameplay ==")
	            gameplay = page_data[location+len("== Gameplay =="):]
	            gameplay = gameplay.split("==")[0]
	            print(gameplay)
	            game_to_gameplay[g] = gameplay

        except wikipedia.exceptions.PageError:
                # print(title)
                print("Error")
                continue


    with open("game_to_gameplay_text.json","w+") as game_file:
    	json.dump(game_to_gameplay,game_file)
