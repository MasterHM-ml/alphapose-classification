import json
import pandas as pd
import numpy as np

def read_file(f:str):
    return open(f, 'r')

def read_json(f) -> dict:
    return json.load(read_file(f))

def read_contents(f):
    data = read_json(f)
    for k in data:
        print(k['image_id'])
