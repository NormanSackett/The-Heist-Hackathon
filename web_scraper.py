import requests
import json
import bs4
import os
from dotenv import load_dotenv

def scraper():
    # api key and url for google scholar web scraper to retrieve training data
    load_dotenv()
    KEY = os.get_key("API_KEY")
    url = "https://api.scrapingdog.com/google_scholar"

    params = {
        "api_key": KEY,
        "query": "\"mri\" + \"issue\"",
        "language": "en",
        "results": 1
    }
    #response = requests.get(url, params=params)
    
    #if response.status_code == 200:
        #data = response.json()
    data = {"search_details":{"query":"\"mri\" + \"issue\"","number_of_results":"About 3,250,000 results"},"profiles":{},"scholar_results":[{"title":"Twenty years of functional MRI: the science and the stories","title_link":"https://www.sciencedirect.com/science/article/pii/S1053811912004223","id":"ZyI7JRdFZ7gJ","displayed_link":"PA Bandettini - Neuroimage, 2012 - Elsevier","snippet":"… ► Functional MRI began in 1991 and has been extremely successful in the past 20 years. ► This special issue highlights some of the major fMRI method developments over the past …","authors":[{"name":"PA Bandettini","link":"https://scholar.google.com/citations?user=X9OdRnYAAAAJ&hl=en&num=1&oi=sra","author_id":"X9OdRnYAAAAJ","scrapingdog_link":"https://api.scrapingdog.com/google_scholar/author?author_id=X9OdRnYAAAAJ&api_key=68e1780dc54b268365d014fa"}],"inline_links":{"versions":{"total":"All 7 versions","link":"https://scholar.google.com/scholar?cluster=13287665191291134567&hl=en&num=1&as_sdt=0,26","cluster_id":"13287665191291134567"},"cited_by":{"total":"Cited by 341","link":"https://scholar.google.com/scholar?cites=13287665191291134567&as_sdt=5,26&sciodt=0,26&hl=en&num=1"},"related_pages_link":"https://scholar.google.com/scholar?q=related:ZyI7JRdFZ7gJ:scholar.google.com/&scioq=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26"}}],"related_searches":[],"pagination":{"current":1,"page_no":{"1":"https://www.scholar.google.com/scholar?start=1&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","2":"https://www.scholar.google.com/scholar?start=2&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","3":"https://www.scholar.google.com/scholar?start=3&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","4":"https://www.scholar.google.com/scholar?start=4&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","5":"https://www.scholar.google.com/scholar?start=5&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","6":"https://www.scholar.google.com/scholar?start=6&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","7":"https://www.scholar.google.com/scholar?start=7&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","8":"https://www.scholar.google.com/scholar?start=8&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26","9":"https://www.scholar.google.com/scholar?start=9&q=%22mri%22+%2B+%22issue%22&hl=en&num=1&as_sdt=0,26"}},"scrapingdog_pagination":{"current":1,"page_no":{"1":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=0","2":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=10","3":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=20","4":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=30","5":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=40","6":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=50","7":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=60","8":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=70","9":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=80","10":"https://api.scrapingdog.com/scholar?api_key=68e1780dc54b268365d014fa&q=\"mri\"+++\"issue\"&page=90"}}}
    print(data["scholar_results"][0]["title_link"])
    #else:
        #print(f"Request failed with status code: {response.status_code}")

scraper()