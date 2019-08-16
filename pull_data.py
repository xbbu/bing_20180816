import pandas as pd
import csv
import urllib
import requests

## pull data from URL link
def get_csv_from_url(url_link):
    """ get csv from url link file
    """
    web_page = urllib.request.urlopen(url_link).read().decode('utf-8')
    data_reader = web_page.split('\n')

    vert_data = [ii.split(',') for ii in data_reader]
    Ncols = len(vert_data[0])
    vert_data = [ii for ii in vert_data if len(ii)==Ncols]

    df = pd.DataFrame(vert_data[1:], columns=vert_data[0])

    return df
