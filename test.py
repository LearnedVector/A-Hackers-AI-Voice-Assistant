import json
from time import sleep
import requests

from client import Client

c = Client()

print(c._ask_input('hey there')['response'])

