import requests 
from datetime import datetime
import logging
import time
import pickle

def get_nextbike_locations ():
    
    # request data
    URL = "https://gbfs.nextbike.net/maps/gbfs/v1/nextbike_bn/en/free_bike_status.json"

    # sending get request and saving the response as response object 
    response = requests.get(url = URL) 
    try:
        r = response.json()
        nextbikes = []
        for i in range(len(r['data']['bikes'])):
            try:
                bike_id = int(r['data']['bikes'][i]['bike_id'])
            except:
                # single bike have no ID (?!); skip these bikes
                continue
            
            lat = r['data']['bikes'][i]['lat']
            lon = r['data']['bikes'][i]['lon']
            nextbikes.append([bike_id, datetime.now(), lat,lon])
        return nextbikes
    except Exception:
        logging.exception("message")
 
bike_locs = {}

while datetime.now() < datetime(2022, 5, 1, 23, 59, 59):
    bike_locs[datetime.now()] = get_nextbike_locations()
    time.sleep(900)
    with open('nextbike.pickle', 'wb') as handle:
        pickle.dump(bike_locs, handle, protocol=pickle.HIGHEST_PROTOCOL)