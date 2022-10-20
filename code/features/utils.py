import pandas as pd
import json
import networkx as nx
import re
from openwpm_utils import domain as du
from six.moves.urllib.parse import urlparse, parse_qs
from sklearn import preprocessing
from yaml import load, dump
import numpy as np
import traceback
from Levenshtein import distance
import tldextract


import graph as gs
import base64
import hashlib

from logger import LOGGER


def extract_data(data, key=None):
    """
    Function to extract data from a dictionary.

    Args:
      data: dictionary
      key: key to extract

    Returns:
      value: value of key

    """

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                for d in extract_data(v, k):
                    yield d
            else:
                yield {k: v}
    elif isinstance(data, list):
        for d in data:
            for d in extract_data(d, key):
                yield d
    elif data is not None:
        yield {key: data}
    else:
        pass


def get_response_content(content_hash, ldb):
    """
    Function to get the content of a response.

    Args:
      content_hash: hash of the response
      ldb: Content LDB

    Returns:
      content: content of the response

    """
    try:
        content = ldb.Get(content_hash.encode('utf-8'))
        jsonData = content.decode('utf-8')
        return json.loads(jsonData)
    except Exception as e:
        return None

def get_domain(url):

    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        #traceback.print_exc()
        return None

def parse_url_arguments(url):
    """
    Function to parse the arguments of a URL.

    Args:
      url: URL

    Returns:
      arguments: arguments of the URL

    """

    parsed = urlparse.urlparse(url)
    sep1 = parse_qs(parsed.query)
    return {**sep1}


def has_ad_keyword(node, G):
    """
    Function to check if a node URL xhas an ad keyword.

    Args:
      node: URL of node
      G: networkX representation of graph
    Returns:
      has_ad_keyword: binary value showing if node URL has ad keyword
    """

    keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                   "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",
                   "delivery", "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc", "google_afs"]
    has_ad_keyword = 0
    node_type = G.nodes[node]['type']
    if node_type != "Element" and node_type != "Storage":
        for key in keyword_raw:
            key_matches = [m.start() for m in re.finditer(key, node, re.I)]
            for key_match in key_matches:
                has_ad_keyword = 1
                break
            if has_ad_keyword == 1:
                break
    return has_ad_keyword


def ad_keyword_ascendants(node, G):
    """
    Function to check if any ascendant of a node has an ad keyword.

    Args:
      node: URL of node
      G: networkX representation of graph
    Returns:
      ascendant_has_ad_keyword: binary value showing if ascendant has ad keyword
    """

    keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                   "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban", "delivery",
                   "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc", "google_afs"]

    ascendant_has_ad_keyword = 0
    ascendants = nx.ancestors(G, node)
    for ascendant in ascendants:
        try:
            node_type = nx.get_node_attributes(G, 'type')[ascendant]
            if node_type != "Element" and node_type != "Storage":
                for key in keyword_raw:
                    key_matches = [m.start()
                                   for m in re.finditer(key, ascendant, re.I)]
                    for key_match in key_matches:
                        ascendant_has_ad_keyword = 1
                        break
                    if ascendant_has_ad_keyword == 1:
                        break
            if ascendant_has_ad_keyword == 1:
                break
        except:
            continue
    return ascendant_has_ad_keyword


def find_modified_storage(df_target):
    """
    Function to find modified edges -- if a storage element is set by Node 1 and modified (set again) by Node 2,
    there will be an edge from Node 1 to Node 2.

    Args:
      df_target: DataFrame representation of all storage sets, for a particular storage element.
    Returns:
      df_modedges: DataFrame representation of modified edges.
    """

    df_modedges = pd.DataFrame()
    df_copy = df_target.copy().sort_values(by=['time_stamp'])
    df_copy = df_copy.reset_index()
    set_node = df_copy.iloc[[0]][['src', 'dst']]
    modify_nodes = df_copy.drop([0], axis=0)[['src', 'dst']]

    if len(modify_nodes) > 0:
        df_merged = pd.merge(set_node, modify_nodes, on='dst')
        df_modedges = df_merged[['src_x', 'src_y', 'dst']].drop_duplicates()
        df_modedges.columns = ['src', 'dst', 'attr']
        df_modedges = df_modedges.groupby(['src', 'dst'])[
            'attr'].apply(len).reset_index()

    return df_modedges


def get_cookieval(attr):
    """
    Function to extract cookie value.

    Args:
      attr: attributes of cookie node
    Returns:
      name: cookie value
    """

    try:
        attr = json.loads(attr)
        if 'value' in attr:
            return attr['value']
        else:
            return None
    except:
        return None


def get_cookiename(attr):
    """
    Function to extract cookie name.

    Args:
      attr: attributes of cookie node
    Returns:
      name: cookie name
    """

    try:
        attr = json.loads(attr)
        if 'name' in attr:
            return attr['name']
        else:
            return None
    except:
        return None


def get_redirect_depths(df_graph):
    """
    Function to extract redirect depths of every node in the graph.

    Args:
      df_graph: DataFrame representation of graph
    Returns:
      dict_redict: dictionary of redirect depths for each node
    """

    dict_redirect = {}

    try:

        http_status = [300, 301, 302, 303, 307, 308]
        http_status = http_status + [str(x) for x in http_status]
        df_redirect = df_graph[df_graph['response_status'].isin(http_status)]

        G_red = gs.build_networkx_graph(df_redirect)

        for n in G_red.nodes():
            dict_redirect[n] = 0
            dfs_edges = list(nx.edge_dfs(
                G_red, source=n, orientation='reverse'))
            ct = 0
            depths = []
            if len(dfs_edges) == 1:
                dict_redirect[n] = 1
            if len(dfs_edges) >= 2:
                ct += 1
                for i in range(1, len(dfs_edges)):
                    if dfs_edges[i][1] != dfs_edges[i-1][0]:
                        depths.append(ct)
                        ct = 1
                    else:
                        ct += 1
                depths.append(ct)
                if len(depths) > 0:
                    dict_redirect[n] = max(depths)

        return dict_redirect

    except Exception as e:
        return dict_redirect


def find_urls(df):
    """
    Function to get set of URLs on a site.

    Args:
      df: DataFrame representation of all edges.
    Returns:
      all_urls: List of URLs.
    """

    src_urls = df['src'].tolist()
    dst_urls = df['dst'].tolist()
    all_urls = list(set(src_urls + dst_urls))
    return all_urls


def check_full_cookie(cookie_value, dest):
    """
    Function to check if a cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a cookie value exists in a URL.
    """

    return True if len([item for item in cookie_value if item in dest and len(item) > 3]) > 0 else False


def check_partial_cookie(cookie_value, dest):
    """
    Function to check if a partial cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a partial cookie value exists in a URL.
    """

    for value in cookie_value:
        split_cookie = re.split(
            r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', value)
        return True if len([item for item in split_cookie if item in dest and len(item) > 3]) > 0 else False
    return False


def check_base64_cookie(cookie_value, dest):
    """
    Function to check if a base64 encoded cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a base64 encoded cookie value exists in a URL.
    """

    return True if len([item for item in cookie_value if base64.b64encode(item.encode('utf-8')).decode('utf8') in dest and len(item) > 3]) > 0 else False


def check_md5_cookie(cookie_value, dest):
    """
    Function to check if a MD5 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a MD5 hashed cookie value exists in a URL.
    """

    return True if len([item for item in cookie_value if hashlib.md5(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_sha1_cookie(cookie_value, dest):
    """
    Function to check if a SHA1 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a SHA1 hashed cookie value exists in a URL.
    """

    return True if len([item for item in cookie_value if hashlib.sha1(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_full_cookie_set(cookie_value, dest):
    """
    Function to check if a cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (cookie_value in dest):
        return True
    else:
        return False


def check_partial_cookie_set(cookie_value, dest):
    """
    Function to check if a partial cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a partial cookie value exists in a URL.
    """

    split_cookie = re.split(
        r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', cookie_value)
    for item in split_cookie:
        if len(item) > 3 and item in dest:
            return True
    return False


def check_base64_cookie_set(cookie_value, dest):
    """
    Function to check if a base64 encoded cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a base64 encoded cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (base64.b64encode(cookie_value.encode('utf-8')).decode('utf8') in dest):
        return True
    else:
        return False


def check_md5_cookie_set(cookie_value, dest):
    """
    Function to check if a MD5 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a MD5 hashed cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (hashlib.md5(cookie_value.encode('utf-8')).hexdigest() in dest):
        return True
    else:
        return False


def check_sha1_cookie_set(cookie_value, dest):
    """
    Function to check if a SHA1 hashed cookie value exists in a URL.

    Args:
      cookie_value: Cookie value
      dest: URL
    Returns:
      Binary value showing whether a SHA1 hashed cookie value exists in a URL.
    """

    if (len(cookie_value) > 3) and (hashlib.sha1(cookie_value.encode('utf-8')).hexdigest() in dest):
        return True
    else:
        return False


def check_cookie_presence(http_attr, dest):

    check_value = False

    try:
        http_attr = json.loads(http_attr)

        for item in http_attr:
            if 'Cookie' in item[0]:
                cookie_pairs = item[1].split(';')
                for cookie_pair in cookie_pairs:
                    cookie_value = cookie_pair.strip().split('=')[1:]
                    full_cookie = check_full_cookie(cookie_value, dest)
                    partial_cookie = check_partial_cookie(cookie_value, dest)
                    base64_cookie = check_base64_cookie(cookie_value, dest)
                    md5_cookie = check_md5_cookie(cookie_value, dest)
                    sha1_cookie = check_sha1_cookie(cookie_value, dest)
                    check_value = check_value | full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
                    if check_value:
                        return check_value
    except:
        check_value = False
    return check_value


def find_indirect_edges(G, df_graph):
    """
    Function to extract shared information edges, used for dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
    Returns:
      df_edges: DataFrame representation of shared information edges.
    """

    df_edges = pd.DataFrame()

    try:

        storage_set = df_graph[(df_graph['action'] == 'set') |
                               (df_graph['action'] == 'set_js') | (df_graph['action']
                                                                   == 'set_storage_js')][['src', 'dst']]
        storage_get = df_graph[(df_graph['action'] == 'get') |
                               (df_graph['action'] == 'get_js') | (df_graph['action']
                                                                   == 'get_storage_js')][['src', 'dst']]

        # Nodes that set to nodes that get
        df_merged = pd.merge(storage_set, storage_get, on='dst')
        df_get_edges = df_merged[['src_x', 'src_y', 'dst']].drop_duplicates()
        if len(df_get_edges) > 0:
            df_get_edges.columns = ['src', 'dst', 'attr']
            df_get_edges['cookie'] = df_get_edges['attr']
            df_get_edges = df_get_edges.groupby(
                ['src', 'dst'])['attr'].apply(len).reset_index()
            df_get_edges['type'] = 'set_get'
            df_edges = pd.concat([df_edges, df_get_edges], ignore_index=True)

        # Nodes that set to nodes that modify
        all_storage_set = df_graph[(df_graph['action'] == 'set') |
                                   (df_graph['action'] == 'set_js') | (
                                       df_graph['action'] == 'set_storage_js')
                                   | (df_graph['action'] == 'remove_storage_js')]
        df_modified_edges = all_storage_set.groupby(
            'dst').apply(find_modified_storage)
        if len(df_modified_edges) > 0:
            df_modified_edges['type'] = 'set_modify'
            df_edges = pd.concat(
                [df_edges, df_modified_edges], ignore_index=True)

        # Nodes that set to URLs with cookie value
        df_set_url_edges = pd.DataFrame()
        df_cookie_set = df_graph[(df_graph['action'] == 'set') |
                                 (df_graph['action'] == 'set_js')].copy()
        df_cookie_set['cookie_val'] = df_cookie_set['attr'].apply(
            get_cookieval)
        cookie_values = list(
            set(df_cookie_set[~df_cookie_set['cookie_val'].isnull()]['cookie_val'].tolist()))

        df_nodes = df_graph[(df_graph['graph_attr'] == 'Node') &
                            ((df_graph['type'] == 'Request') |
                            (df_graph['type'] == 'Script') |
                            (df_graph['type'] == 'Document'))]['name']
        urls = df_nodes.tolist()
        check_set_value = False

        for dest in urls:
            for cookie_value in cookie_values:
                full_cookie = check_full_cookie_set(cookie_value, dest)
                partial_cookie = check_partial_cookie_set(cookie_value, dest)
                base64_cookie = check_base64_cookie_set(cookie_value, dest)
                md5_cookie = check_md5_cookie_set(cookie_value, dest)
                sha1_cookie = check_sha1_cookie_set(cookie_value, dest)
                check_set_value = full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
                if check_set_value:
                    src = df_cookie_set[df_cookie_set['cookie_val']
                                        == cookie_value]['src'].iloc[0]
                    dst = dest
                    attr = 1
                    df_set_url_edges = pd.concat([df_set_url_edges, pd.DataFrame.from_records(
                        [{'src': src, 'dst': dst, 'attr': attr}])], ignore_index=True)

        if len(df_set_url_edges) > 0:
            df_set_url_edges = df_set_url_edges.groupby(
                ['src', 'dst'])['attr'].apply(len).reset_index()
            df_set_url_edges['type'] = 'set_url'
            df_edges = pd.concat(
                [df_edges, df_set_url_edges], ignore_index=True)

        # Nodes that get to URLs with cookie value
        df_http_requests = df_graph[(df_graph['reqattr'] != 'CS') & (
            df_graph['src'] != 'N/A') & (df_graph['action'] != 'CS') & (df_graph['graph_attr'] != 'EdgeWG')]
        df_http_requests_merge = pd.merge(left=df_http_requests, right=df_http_requests, how='inner', left_on=[
                                          'visit_id', 'dst'], right_on=['visit_id', 'src'])
        df_http_requests_merge = df_http_requests_merge[df_http_requests_merge['reqattr_x'].notnull(
        )]

        if len(df_http_requests_merge):
            df_http_requests_merge['cookie_presence'] = df_http_requests_merge.apply(
                axis=1,
                func=lambda x: check_cookie_presence(
                    x['reqattr_x'], x['dst_y'])
            )

            df_get_url_edges = df_http_requests_merge[df_http_requests_merge['cookie_presence'] == True][[
                'src_x', 'dst_y', 'attr_x']]
            if len(df_get_url_edges) > 0:
                df_get_url_edges.columns = ['src', 'dst', 'attr']
                df_get_url_edges = df_get_url_edges.groupby(
                    ['src', 'dst'])['attr'].apply(len).reset_index()
                df_get_url_edges['type'] = 'get_url'
                df_edges = pd.concat(
                    [df_edges, df_get_url_edges], ignore_index=True)

    except Exception as e:
        LOGGER.exception(
            "An error occurred when extracting shared information edges.")
        return df_edges

    return df_edges


def check_base64_cookie(cookie_value, dest):
    return True if len([item for item in cookie_value if base64.b64encode(item.encode('utf-8')).decode('utf8') in dest and len(item) > 3]) > 0 else False


def check_md5_cookie(cookie_value, dest):
    return True if len([item for item in cookie_value if hashlib.md5(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_sha1_cookie(cookie_value, dest):
    return True if len([item for item in cookie_value if hashlib.sha1(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_full_cookie_set(cookie_value, dest):
    if (len(cookie_value) > 3) and (cookie_value in dest):
        return True
    else:
        return False


def check_partial_cookie_set(cookie_value, dest):
    split_cookie = re.split(
        r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', cookie_value)
    for item in split_cookie:
        if len(item) > 3 and item in dest:
            return True
    return False


def check_base64_cookie_set(cookie_value, dest):
    if (len(cookie_value) > 3) and (base64.b64encode(cookie_value.encode('utf-8')).decode('utf8') in dest):
        return True
    else:
        return False


def check_md5_cookie_set(cookie_value, dest):
    if (len(cookie_value) > 3) and (hashlib.md5(cookie_value.encode('utf-8')).hexdigest() in dest):
        return True
    else:
        return False


def check_sha1_cookie_set(cookie_value, dest):
    if (len(cookie_value) > 3) and (hashlib.sha1(cookie_value.encode('utf-8')).hexdigest() in dest):
        return True
    else:
        return False


def check_cookie_presence(http_attr, dest):

    check_value = False

    try:
        http_attr = json.loads(http_attr)

        for item in http_attr:
            if 'Cookie' in item[0]:
                cookie_pairs = item[1].split(';')
                for cookie_pair in cookie_pairs:
                    cookie_value = cookie_pair.strip().split('=')[1:]
                    full_cookie = check_full_cookie(cookie_value, dest)
                    partial_cookie = check_partial_cookie(cookie_value, dest)
                    base64_cookie = check_base64_cookie(cookie_value, dest)
                    md5_cookie = check_md5_cookie(cookie_value, dest)
                    sha1_cookie = check_sha1_cookie(cookie_value, dest)
                    check_value = check_value | full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
                    if check_value:
                        return check_value
    except:
        check_value = False
    return check_value


def compare_with_obfuscation(cookie_value, param_value, threshold):

    if (cookie_value == param_value or distance(cookie_value, param_value) < threshold):
        return True
    encoded_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = base64.b64encode(
        cookie_value.encode('utf-8')).decode('utf8')
    if (encoded_value == param_value):
        return True

    return False


def check_in_header(cookie_value, headers, threshold):

    headers = [x for x in headers if len(x) > 5]
    for header in headers:
        difference = abs(len(cookie_value) - len(header))
        if difference > 10:
            continue
        offset = len(cookie_value) - difference if len(cookie_value) > len(
            header) else len(header)-difference
        for i in range(0, difference+1):
            if len(header) > len(cookie_value):
                if (compare_with_obfuscation(cookie_value, header[i:i+offset], threshold)):
                    return True
            else:
                if (compare_with_obfuscation(cookie_value[i:i+offset], headers, threshold)):
                    return True
    return False


def check_in_body(cookie_value, body):

    if (cookie_value in body):
        return True
    encoded_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value in body):
        return True

    encoded_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value in body):
        return True

    encoded_value = base64.b64encode(
        cookie_value.encode('utf-8')).decode('utf8')
    if (encoded_value in body):
        return True
    return False


def check_if_cookie_value_exists_post_body(cookie_key, cookie_value, post_body, threshold):

    cookie_value = str(cookie_value)
    if (len(cookie_value) <= 5):
        return False, None
    if check_in_body(cookie_value, post_body):
        return "True", "out"
    return False, None


def check_if_cookie_value_exists_header(cookie_key, cookie_value, reqheader_values, respheader_values, threshold):

    cookie_value = str(cookie_value)
    if (len(cookie_value) <= 5):
        return False, None

    # if cookie_key in reqheader_values:
    #   return True, "out"
    # if cookie_key in respheader_values:
    #   return True, "in"

    if check_in_header(cookie_value, reqheader_values, threshold):
        return True, "out"
    if check_in_header(cookie_value, respheader_values, threshold):
        return True, "in"
    return False, None


def check_if_cookie_value_exists_content(cookie_key, cookie_value, json_data, threshold):

    cookie_value = str(cookie_value)
    if (len(cookie_value) <= 5):
        return False, None

    try:
        for data in json_data:
            key = list(data.keys())[0]
            if (cookie_value == data[key]):
                return True, key
            if not isinstance(data[key], str):
                continue
            if(len(data[key]) < 5):
                continue
            difference = abs(len(cookie_value) - len(data[key]))
            if difference > 10:
                continue
            offset = len(cookie_value) - difference if len(cookie_value) > len(
                data[key]) else len(data[key])-difference
            for i in range(0, difference+1):
                if len(data[key]) > len(cookie_value):
                    if (compare_with_obfuscation(cookie_value, data[key][i:i+offset], threshold)):
                        return True, key
                else:
                    if (compare_with_obfuscation(cookie_value[i:i+offset], data[key][0], threshold)):
                        return True, key
    except:
        return False, None
    return False, None


def check_if_cookie_value_exists(cookie_key, cookie_value, param_dict, threshold):

    cookie_value = str(cookie_value)
    if (len(cookie_value) <= 5):
        return False, None

    for key in param_dict:
        if (cookie_key == param_dict[key][0]):
            return True, key
        if(len(param_dict[key][0]) < 5):
            continue
        difference = abs(len(cookie_value) - len(param_dict[key][0]))
        if difference > 10:
            continue
        offset = len(cookie_value) - difference if len(cookie_value) > len(
            param_dict[key][0]) else len(param_dict[key][0])-difference
        for i in range(0, difference+1):
            if len(param_dict[key][0]) > len(cookie_value):
                if (compare_with_obfuscation(cookie_value, param_dict[key][0][i:i+offset], threshold)):
                    return True, key
            else:
                if (compare_with_obfuscation(cookie_value[i:i+offset], param_dict[key][0], threshold)):
                    return True, key

    return False, None


def get_header_values(header_string):
    """
    Returns a list of header values from a header string
    
    Args:
        header_string (str): The header string to parse

    Returns:
        list: A list of header values
    """

    values = []
    try:
        headers = json.loads(header_string)
        for header in headers:
            if (str(header[0].lower()) != "cookie") and (str(header[0].lower()) != "set-cookie"):
                values.append(header[1])
    except Exception as e:
        #print(e, header_string)
        # traceback.print_exc()
        return values
    return values


def get_ls_name(name):
    """
    Get the name of the local storage
    
    Args:
        name (str): The name of the local storage

    Returns:
        str: The name of the local storage
    """

    try:
        parts = name.split("|$$|")
        if len(parts) == 3:
            return name.rsplit("|$$|", 1)[0]
    except:
        return name
    return name


def find_exfiltrations(df_graph, ldb):
    """
    Find exfiltrations in the graph
    
    Args:
        df_graph (pandas.DataFrame): The graph dataframe
        ldb (leveldb.LevelDB): The LevelDB database

    Returns:
        pandas.DataFrame: The exfiltrations dataframe
    """

    find_exfiltrations.df_edges = pd.DataFrame(
        columns=['visit_id', 'src', 'dst', 'attr', 'time_stamp', 'direction'])

    try:

        df_cookie_set = df_graph[(df_graph['action'] == 'set') |
                                 (df_graph['action'] == 'set_js')].copy()
        df_cookie_set['cookie_value'] = df_cookie_set['attr'].apply(
            get_cookieval)
        df_cookie_set = df_cookie_set[[
            'dst', 'cookie_value']].drop_duplicates()
        df_cookie_set = df_cookie_set.rename(columns={'dst': 'cookie_key'})
        cookie_names = df_cookie_set['cookie_key'].unique().tolist()

        # Check LS with same cookie name
        df_ls_set = df_graph[(df_graph['action'] == 'set_storage_js')].copy()
        df_ls_set['split_name'] = df_ls_set['dst'].apply(get_ls_name)
        df_ls_set = df_ls_set[df_ls_set['split_name'].isin(cookie_names)]
        df_ls_set['cookie_value'] = df_ls_set['attr'].apply(get_cookieval)
        df_ls_set = df_ls_set[['dst', 'cookie_value']].drop_duplicates()
        df_ls_set = df_ls_set.rename(columns={'dst': 'cookie_key'})

        df_requests = df_graph[(df_graph['graph_attr'] == 'Node') &
                               ((df_graph['type'] == 'Request') |
                                (df_graph['type'] == 'Script') |
                                (df_graph['type'] == 'Document'))]

        df_headers = df_graph[(df_graph['reqattr'].notnull())
                              & (df_graph['reqattr'] != "N/A")]

        df_post_bodies = df_graph[(df_graph['post_body'].notnull()) | (
            df_graph['post_body_raw'].notnull())]

        df_content = df_graph[(df_graph['content_hash'].notnull()) | (
            df_graph['content_hash'] != 'N/A')]
        df_content['content'] = df_content['content_hash'].apply(
            get_response_content, ldb=ldb)
        df_content = df_content[df_content['content'].notnull()]

        def process_cookie(cookie_row):

            cookie_key = cookie_row['cookie_key']
            cookie_value = cookie_row['cookie_value']
            cookie_key_stripped = cookie_key.split("|$$|")[0].strip()

            def process_request(row):

                header_cookies = {}
                url_parameters = parse_url_arguments(row['name'])
                values_dict = {**header_cookies, **url_parameters}
                exists, key = check_if_cookie_value_exists(
                    cookie_key_stripped, cookie_value, values_dict, 2)
                if exists:
                    find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id': row['visit_id'], 'src': cookie_key, 'dst': row['name'], 'dst_domain': row['domain'],
                                                                                      'attr': cookie_value, 'time_stamp': row['time_stamp'], 'direction': 'out', 'type': 'url'}, ignore_index=True)

            def process_header(row):

                reqheader_values = get_header_values(row['reqattr'])
                respheader_values = get_header_values(row['respattr'])

                if (len(reqheader_values) > 0) & (len(respheader_values) > 0):
                    exists, direction = check_if_cookie_value_exists_header(
                        cookie_key_stripped, cookie_value, reqheader_values, respheader_values, 2)
                    if exists:
                        if direction == "in":
                            find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id': row['visit_id'], 'src': cookie_key, 'dst': row['dst'], 'dst_domain': row['dst_domain'],
                                                                                              'attr': cookie_value, 'time_stamp': row['time_stamp'], 'direction': 'in', 'type': 'header'}, ignore_index=True)
                        else:
                            find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id': row['visit_id'], 'src': cookie_key, 'dst': row['dst'], 'dst_domain': row['dst_domain'],
                                                                                              'attr': cookie_value, 'time_stamp': row['time_stamp'], 'direction': 'out', 'type': 'header'}, ignore_index=True)

            def process_post_bodies(row):

                exists = False
                body_value = ""
                if (row['post_body']) and (row['post_body'] != "CS"):
                    body_value = row['post_body']
                elif row['post_body_raw'] and (row['post_body_raw'] != "CS"):
                    try:
                        body = json.loads(row['post_body_raw'])
                        if len(body) > 0:
                            body_value = body[0][1]
                            body_value = base64.b64decode(body_value).decode()
                    except:
                        traceback.print_exc()
                        body_value = ""
                if len(body_value) > 1:
                    exists, direction = check_if_cookie_value_exists_post_body(
                        cookie_key_stripped, cookie_value, body_value, 2)
                if exists:
                    find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id': row['visit_id'], 'src': cookie_key, 'dst': row['dst'], 'dst_domain': row['dst_domain'],
                                                                                      'attr': cookie_value, 'time_stamp': row['time_stamp'], 'direction': 'out', 'type': 'postbody'}, ignore_index=True)

            def process_response_content(row):

                exists = False
                content = row['content']
                json_data = []
                try:
                    for d in extract_data(content):
                        json_data.append(d)
                except:
                    pass

                if (len(json_data) > 0) & (cookie_value is not None):
                    exists, key = check_if_cookie_value_exists_content(
                        cookie_key_stripped, cookie_value, json_data, 2)
                    if exists:
                        find_exfiltrations.df_edges = find_exfiltrations.df_edges.append({'visit_id': row['visit_id'], 'src': cookie_key, 'dst': row['dst'], 'dst_domain': row['dst_domain'],
                                                                                          'attr': cookie_value, 'time_stamp': row['time_stamp'], 'direction': 'in', 'type': 'content'}, ignore_index=True)

            df_requests.apply(process_request, axis=1)
            df_headers.apply(process_header, axis=1)
            df_post_bodies.apply(process_post_bodies, axis=1)
            if len(df_content) > 0:
                df_content.apply(process_response_content, axis=1)
        df_cookie_set.apply(process_cookie, axis=1)
        df_ls_set.apply(process_cookie, axis=1)

    except Exception as e:
        print("Error in exfiltration edges")
        traceback.print_exc()
        return find_exfiltrations.df_edges

    return find_exfiltrations.df_edges
