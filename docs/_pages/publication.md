---
permalink: /about/
layout: minimal
title: "Publication Details"
---

# Abstract

As third-party cookie blocking is becoming the norm in browsers, advertisers and trackers have started to use first-party cookies for tracking. We conduct a differential measurement study on 10K websites with third-party cookies allowed and blocked. This study reveals that first-party cookies are used to store and exfiltrate identifiers to known trackers even when third-party cookies are blocked. 

As opposed to third-party cookie blocking, outright first-party cookie blocking is not practical because it would result in major functionality breakage. We propose CookieGraph, a machine learning-based approach that can accurately and robustly detect first-party tracking cookies. CookieGraph detects first-party tracking cookies with 90.20% accuracy, outperforming the state-of-the-art CookieBlock approach by 17.75%. We show that CookieGraph is fully robust against cookie name manipulation while CookieBlock's accuracy drops by 15.68%. While blocking all first-party cookies results in major breakage on 32% of the sites with SSO logins, and CookieBlock reduces it to 10%, we show that CookieGraph does not cause any major breakage on these sites. 

Our deployment of CookieGraph shows that first-party tracking cookies are used on 93.43% of the 10K websites. We also find that first-party tracking cookies are set by fingerprinting scripts. The most prevalent first-party tracking cookies are set by major advertising entities such as Google, Facebook, and TikTok.