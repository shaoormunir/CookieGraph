---
layout: page
title: Dataset Information
subtitle: ""
---

We ran CookieGraph on 10k websites (top 1k websites, and 9k sites uniformly randomly sampled from 1k to 100k). We provide here all the tracking first-party cookies detected by CookieGraph, along with their setter scripts, and the different domains to which they are exfiltrated.

## Dataset Format

The dataset is in the form of a JSON file. The structure of the JSON file is as follows:

``` JSON
{ 
  "https://www.example.com": {
    "cookie_key": {
      "setter_scripts": [
        "http://www.example.com/setter1.js",
        "http://www.example.com/setter2.js"],
      "exfiltration_endpoints": [
        "http://www.example.com/endpoint1",
        "http://www.example.com/endpoint2"
      ]
    }
  }
}
```

The raw JSON file containing the dataset can be found [here](https://raw.githubusercontent.com/shaoormunir/CookieGraph/main/data/dataset.json).

## Conversion to uBlock filterlist

The script ```convert_to_ublock_filterlist.py```, found [here](https://raw.githubusercontent.com/shaoormunir/CookieGraph/main/data/convert_to_ublock_filiterlist.py), can be used to convert the dataset to a uBlock filterlist. The filterlist makes use of cookie-remover.js of uBlock to identify tracking cookies per site base. You can run the script using the following command:

```SHELL
python3 convert_to_ublock_filterlist.py -i input.json -o output.txt
```

Each entry in the filterlist follows the following format:

```
website-name##+js(cookie-remover.js, cookie_name)