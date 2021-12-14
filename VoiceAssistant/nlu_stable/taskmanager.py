import datetime
import glob
import json
import multiprocessing
import os
import math
import random
import socket

import lxml
import requests
import subprocess
from time import sleep
from urllib.request import urlopen

import wikipedia
import ipinfo
from bs4 import BeautifulSoup as soup
from pytube import YouTube
from youtubesearchpython import VideosSearch
from word2number import w2n

from VoiceAssistant.nlu_stable.etc.memory import MemoryUnit
from VoiceAssistant.nlu_stable.etc.qna_parser import Parser
from VoiceAssistant.nlu_stable.client_config import *


p = Parser()
jokelist = []



class TaskManager(MemoryUnit):

    def __init__(self):
        super().__init__()
        self.event = False


    def ocr_read(self):
        # observe_direction()
        # text = read_text()
        # return text
        pass

    def weather(self):
        ip_address = socket.gethostbyname(hostname)
        handler = ipinfo.getHandler(ip_location_api_token)
        details = handler.getDetails(ip_address)
        city_name = details.city
        # base_url variable to store url
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
         
        # complete_url variable to store
        # complete url address
        complete_url = base_url + "appid=" + open_weather_api_token + "&q=" + city_name
         
        # get method of requests module
        # return response object
        response = requests.get(complete_url)
         
        # json method of response object
        # convert json format data into
        # python format data
        x = response.json()
         
        # Now x contains list of nested dictionaries
        # Check the value of "cod" key is equal to
        # "404", means city is found otherwise,
        # city is not found
        if x["cod"] != "404":
         
            # store the value of "main"
            # key in variable y
            y = x["main"]
         
            # store the value corresponding
            # to the "temp" key of y
            current_temperature = str(math.ceil((int(y["temp"]) - 273)))+ " Celcius"
         
            # store the value corresponding
            # to the "pressure" key of y
            
         
            # store the value corresponding
            # to the "humidity" key of y
            current_humidity = y["humidity"]
         
            # store the value of "weather"
            # key in variable z
            z = x["weather"]
         
            # store the value corresponding
            # to the "description" key at
            # the 0th index of z
            weather_description = z[0]["description"]
         
            
            return weather_description, current_temperature, current_humidity
 
        else:
            return "City Not Found", "City Not Found", "City Not Found"

    
    def take_note(self, text):
        """just pass the text to be saved or notted down"""

        self.date = str(datetime.datetime.now().date()) + "%" + str(datetime.datetime.now().hour) + "+" + str(
            datetime.datetime.now().minute) + "}"
        self.file_name = "notes/" + str(self.date).replace(":", "-") + "-note.txt"
        with open(self.file_name, "w") as f:
            f.write(text)
        # subprocess.Popen(["notepad.exe", self.file_name])
   
    def get_note(self, args):
        """
        available args:
            latest : reads latest note
            total : returns num of notes
            yesterday : returns yesterday's note

        """
        self.list_of_files = glob.glob('notes/*')  # * means all if need specific format then *.csv

        if "latest" in args.lower() or "last note" in args.lower():
            self.latest_file = max(self.list_of_files, key=os.path.getctime)
            self.latest_file = str(self.latest_file.replace("notes", ""))

            with open(f"notes{self.latest_file}", "r") as g:
                return g.read()

        elif "total" in args.lower() or "how many" in args.lower():
            return len(self.list_of_files)

        elif "yesterday" in args.lower():
            self.ys = str(datetime.datetime.now().day)
            self.ys = int(self.ys) - 1
            print(self.ys)
            self.mn = datetime.datetime.now().month
            self.yr = datetime.datetime.now().year
            print(f"{self.yr}-{self.mn}-{self.ys}")
            for i in self.list_of_files:
                if f"{self.yr}-{self.mn}-{self.ys}" in i:
                    with open(f"{i}", "r") as re:
                        return re
                else:
                    return "you haven't made any entries yesterday"

    # def get_note_time
    def get_note_time(self, filename, arg="ymd"):
        self.ymd = filename[:"%"]
        self.hour = filename["%":"+"]
        self.minute = filename["+":]

        if arg == "ymd":
            return self.ymd
        elif arg == "hr":
            return self.hour
        elif arg == "min":
            return self.hour

    def news(self, headlines):
        """

        --------------------------------------------------------------------------------------------
        :ARGS: Headlines(str)     [number of headlines you want]

        :PARSING: https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en
        change US in the above link to IN for Indian news, CA for Canada, and so on.

        Keep it just https://news.google.com/rss for dynanimic location selection based on your IP 
        address

        :OUTPUT: returns a list of headlines
        --------------------------------------------------------------------------------------------

        """

        self.nl = []
        try:
            self.int_num = int(w2n.word_to_num(headlines))
            print(self.int_num)
            self.newsurl = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
            self.root = urlopen(self.newsurl)
            self.xmlpage = self.root.read()
            self.root.close()
            self.souppage = soup(self.xmlpage, "xml")
            self.newslist = self.souppage.findAll("item")
            for news in self.newslist[:self.int_num]:
                # speak(news.pubDate.text)
                sleep(1)
                self.nl.append(news.title.text)
            return self.nl

        except Exception as e:
            return f"Looks like something went wrong. Try connecting to internet. {e}"

    
    def wiki(self, query):
        
        """
        
        Get summary of topics from wikipedia.
        Requested args: query(the topic you want to search)

        NOTE: INCREASE sentences=3 TO ANY NUMBER IF REQUIRED,
        HIGHER THE VALUE = LONGER INFO
        SMALLER THE VALUE = LESS INFO AND NOT MUCH USEFULL INFO
        IS RETRIEVED
        
        """
        try:
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return e

    
    def parse_youtube_query(self, query):
        videosSearch = VideosSearch(query, limit = 1)
        link = self.get_youtube_audio(videosSearch.result()['result'][0]['link'])
        return link


    def get_youtube_audio(self, link):
        """
        :INPUT: Youtube video link

        :PROCESS: Downloads the audio of the video only, 
        and saves it to music directory.

        :OUTPUT: Returns nothing, just saves the music at /music dir
        """
        self.yt = YouTube(link)
        self.t = self.yt.streams.filter(only_audio=True)
        self.t[0].download(f"music/{link}")
        print(f"downloaded {link}")
        return link

    """
    def player(self):
        global stop_thread
        url = "https://www.youtube.com/watch?v=svT7uKdNphU"
        video = pafy.new(url)
        best = video.getbest()
        playurl = best.url

        Instance = vlc.Instance()
        player = Instance.media_player_new()
        Media = Instance.media_new(playurl)
        Media.get_mrl()
        player.set_media(Media)
        player.play()
    """

    def play(self, query):
        os.startfile(f"music/{query}")
        self.event = True


    def memorise(self, question, answer):
        """
        :param question:
        :param answer:
        :return:
        """
        self.q = p.parse_question(question)
        self.a = p.parse_answer(answer)
        self.response = self.read_from_db(self.q)
        
        if not self.response:
            self.data_entry(self.q, self.a)
        else:
            return self.response

    def google(self, query):
        """

        :param query:
        :return: google snippet text
        """
        headers = {
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
        }

        html = requests.get(f'https://www.google.com/search?q={query}', headers=headers).text

        self.sp = soup(html, 'lxml')
        try:
            summary = self.sp.select_one('.Uo8X3b+ span').text
            if len(summary) == 0:
                return None
            else:
                return summary

        except:
            try:
                l = self.sp.select(".TrT0Xe")
                self.google_list = []
                for i in l:
                    self.google_list.append(i.text)
                if len(self.google_list) == 0:
                    return None
                else:
                    return ", ".join(self.google_list)
            except:
                return None


    def joke(self):
        """
        return jokes
        """
        f = r"https://official-joke-api.appspot.com/random_ten"
        try:
            data = requests.get(f)
            data = json.loads(data.text)
            if len(jokelist) == 0:
                print("from web")
                for jokes in data:
                    jokelist.append(jokes["setup"]+" "+jokes["punchline"])
                return random.choice(jokelist)
            else:
                print("from storage")
                return random.choice(jokelist)

        except Exception as e:
            return "unable to get jokes right now"
