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