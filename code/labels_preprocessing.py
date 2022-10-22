import argparse
import os 
import sqlite3
import pandas as pd
import tldextract
import json
import requests
import hashlib
from tqdm.auto import tqdm

def get_domain(url):
    """
    Function to extract the domain from a URL.
    
    Args:
        url: URL.
    Returns:
        domain: Domain.
    """
    domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    return domain



def find_common(setters, fname_common, fname_empty):

	hashdict = {}
	others = []
	empty = []
	errors = []
	ct = 0

	print(len(setters))
	for setter in tqdm(setters):
		try:
			print(ct)
			r = requests.get(setter, timeout=10)
			if (len(r.content)) > 0:
				if r.status_code == 200:
					hash_str = hashlib.sha1(r.content).hexdigest()
					if hash_str not in hashdict:
						hashdict[hash_str] = []
					hashdict[hash_str].append(setter)
				else:
					others.append((setter, r.status_code))
			else:
				empty.append(setter)
			
		except Exception as e:
			print(setter, e)
			errors.append(setter)
		ct += 1

		if ct % 100 == 0:
			with open(fname_common, "w") as f:
				f.write(json.dumps(hashdict, indent=4))
			with open(fname_empty, "w") as f:
				f.write(json.dumps({'empty' : empty, 'others' : others, 'errors' : errors}, indent=4))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add arguments for database file, labels file, and the output file
    parser.add_argument('-d', 'database', help='Path to the database file')
    parser.add_argument('-l', 'labels', help='Path to the labels file')
    parser.add_argument('-o', 'output', help='Path to the output file')
    args = parser.parse_args()
    # read the database file

    FNAME = "setters.json"
    OUT = "common.json"
    EMPTY = "empty.json"

    try:
        conn = sqlite3.connect(args.database)
        df_javascript_cookie = pd.read_sql_query("SELECT visit_id, top_level_url, document_url, script_url, operation, symbol FROM javascript where symbol = window.document.cookie", conn)

        df_javascript_cookie['top_level_domain'] = df_javascript_cookie['top_level_url'].apply(get_domain)
        df_javascript_cookie['document_domain'] = df_javascript_cookie['document_url'].apply(get_domain)
        df_javascript_cookie['script_domain'] = df_javascript_cookie['script_url'].apply(get_domain)

        # filter only first-party cookies
        df_javascript_cookie = df_javascript_cookie[df_javascript_cookie['top_level_domain'] == df_javascript_cookie['document_domain']]

        setters = df_javascript_cookie[df_javascript_cookie['operation'] == 'set'].script_url.uqnie().tolist()

        find_common(setters, OUT, EMPTY)

        


    except Exception as e:
        print(e)
        exit(1)
        

