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


def requester(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {
        'User-Agent': user_agent,
    }

    request = urllib.request.Request(url, None, headers)  #The assembled request
    response = urllib.request.urlopen(request)
    data = response.read()  # The data u need
    soup = BeautifulSoup(data, 'lxml')

    return soup


if __name__ == "__main__":

    data = requester("https://gym.openai.com/envs/#atari")
    include_term = "ram"

    game_titles = data.findAll('h3', {"class": "EnvironmentsList-Cell-title"})

    with open("atari_titles.txt", "w+") as atari_file:
        for gt in game_titles:
            if include_term in gt.text:
                print(gt.text)
                atari_file.write(gt.text.split('-')[0] + "\n")
