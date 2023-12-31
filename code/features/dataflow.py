import pandas as pd
import networkx as nx
from .utils import *

from logger import LOGGER


def find_common_name(name):

    parts = name.split("|$$|")
    if len(parts) == 3:
        return name.rsplit("|$$|", 1)[0]
    return name


def get_setter_features(df_graph, df_graph_indirect, node, setter_domain):

    setter_exfil = 0
    setter_redirects_sent = 0
    setter_redirects_rec = 0

    try:
        setter_exfil = len(
            df_graph_indirect[df_graph_indirect['dst_domain'] == setter_domain])

        http_status = [300, 301, 302, 303, 307, 308]

        setter_redirects_sent = len(df_graph[(df_graph['src_domain'] == setter_domain) & (
            df_graph['response_status'].isin(http_status))])
        setter_redirects_rec = len(df_graph[(df_graph['dst_domain'] == setter_domain) & (
            df_graph['response_status'].isin(http_status))])

        setter_features = [setter_exfil,
                           setter_redirects_sent, setter_redirects_rec]
        setter_feature_names = ['setter_exfil',
                                'setter_redirects_sent', 'setter_redirects_rec']

    except:
        setter_features = [setter_exfil,
                           setter_redirects_sent, setter_redirects_rec]
        setter_feature_names = ['setter_exfil',
                                'setter_redirects_sent', 'setter_redirects_rec']
        return setter_features, setter_feature_names

    return setter_features, setter_feature_names


def get_storage_features(df_graph, node):
    """
    Function to extract storage features.

    Args:
      df_graph: DataFrame representation of graph
      node: URL of node
    Returns:
      storage_features: storage feature values
      storage_feature_names: storage feature names
    """

    num_get_storage = 0
    num_set_storage = 0
    num_get_storage_ls = 0
    num_set_storage_ls = 0
    num_ls_gets = 0
    num_ls_sets = 0
    num_ls_gets_js = 0
    num_ls_sets_js = 0
    num_cookieheader_exfil = 0

    try:
        cookie_js_get = df_graph[(df_graph['dst'] == node) & (
            df_graph['action'] == 'get_js')]

        cookie_js_set = df_graph[(df_graph['dst'] == node) & (
            df_graph['action'] == 'set_js')]

        cookie_get = df_graph[(df_graph['src'] == node) &
                              ((df_graph['action'] == 'get') | (df_graph['action'] == 'get_js'))]

        cookie_set = df_graph[(df_graph['src'] == node) &
                              ((df_graph['action'] == 'set') | (df_graph['action'] == 'set_js'))]

        cookie_header = df_graph[(df_graph['dst'] == node) &
                                 (df_graph['action'] == 'get')]

        localstorage_get = df_graph[(df_graph['src'] == node) &
                                    (df_graph['action'] == 'get_storage_js')]

        localstorage_set = df_graph[(df_graph['src'] == node) &
                                    (df_graph['action'] == 'set_storage_js')]

        num_get_storage = len(cookie_get) + len(localstorage_get)
        num_set_storage = len(cookie_set) + len(localstorage_set)
        num_get_storage_js = len(cookie_js_get)
        num_set_storage_js = len(cookie_js_set)

        df_graph_gets = df_graph[(df_graph['action'] == 'get') | (
            df_graph['action'] == 'get_js') | (df_graph['action'] == 'get_storage_js')].copy()
        df_graph_sets = df_graph[(df_graph['action'] == 'set') | (
            df_graph['action'] == 'set_js') | (df_graph['action'] == 'set_storage_js')].copy()
        df_graph_gets['new_dst'] = df_graph_gets['dst'].apply(find_common_name)
        df_graph_sets['new_dst'] = df_graph_sets['dst'].apply(find_common_name)

        num_ls_gets = len(df_graph_gets[df_graph_gets['new_dst'] == node])
        num_ls_sets = len(df_graph_sets[df_graph_sets['new_dst'] == node])

        df_graph_gets_js = df_graph[(df_graph['action'] == 'get_js') | (
            df_graph['action'] == 'get_storage_js')].copy()
        df_graph_sets_js = df_graph[(df_graph['action'] == 'set_js') | (
            df_graph['action'] == 'set_storage_js')].copy()
        df_graph_gets_js['new_dst'] = df_graph_gets_js['dst'].apply(
            find_common_name)
        df_graph_sets_js['new_dst'] = df_graph_sets_js['dst'].apply(
            find_common_name)

        num_ls_gets_js = len(
            df_graph_gets_js[df_graph_gets_js['new_dst'] == node])
        num_ls_sets_js = len(
            df_graph_sets_js[df_graph_sets_js['new_dst'] == node])

        num_cookieheader_exfil = len(cookie_header)

        storage_features = [num_get_storage, num_set_storage, num_get_storage_js, num_set_storage_js, num_ls_gets, num_ls_sets,
                            num_ls_gets_js, num_ls_sets_js, num_cookieheader_exfil]
        storage_feature_names = ['num_get_storage', 'num_set_storage', 'num_get_storage_js', 'num_set_storage_js',
                                 'num_get_storage_ls', 'num_set_storage_ls',
                                 'num_get_storage_ls_js', 'num_set_storage_ls_js', 'num_cookieheader_exfil']

    except:
        storage_features = [num_get_storage, num_set_storage, num_get_storage_ls, num_set_storage_ls, num_ls_gets, num_ls_sets,
                            num_ls_gets_js, num_ls_sets_js, num_cookieheader_exfil]
        storage_feature_names = ['num_get_storage', 'num_set_storage', 'num_get_storage_js', 'num_set_storage_js',
                                 'num_get_storage_ls', 'num_set_storage_ls',
                                 'num_get_storage_ls_js', 'num_set_storage_ls_js', 'num_cookieheader_exfil']

    return storage_features, storage_feature_names


def get_redirect_features(df_graph, node, dict_redirect):
    """
    Function to extract redirect features.

    Args:
      df_graph: DataFrame representation of graph
      node: URL of node
      dict_redirect: dictionary of redirect depths for each node
    Returns:
      redirect_features: redirect feature values
      redirect_feature_names: redirect feature names
    """

    http_status = [300, 301, 302, 303, 307, 308]
    http_status = http_status + [str(x) for x in http_status]

    redirects_sent = df_graph[(df_graph['src'] == node) & (
        df_graph['response_status'].isin(http_status))]
    redirects_rec = df_graph[(df_graph['dst'] == node) & (
        df_graph['response_status'].isin(http_status))]
    num_redirects_sent = len(redirects_sent)
    num_redirects_rec = len(redirects_rec)

    max_depth_redirect = 0
    if node in dict_redirect:
        max_depth_redirect = dict_redirect[node]

    redirect_features = [num_redirects_sent,
                         num_redirects_rec, max_depth_redirect]
    redirect_feature_names = ['num_redirects_sent',
                              'num_redirects_rec', 'max_depth_redirect']

    return redirect_features, redirect_feature_names


def get_request_flow_features(G, df_graph, node):
    """
    Function to extract request flow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
      node: URL of node
    Returns:
      rf_features: request flow feature values
      rf_feature_names: request flow feature names
    """

    requests_sent = df_graph[(df_graph['src'] == node) & (df_graph['reqattr'].notnull()) & (
        df_graph['reqattr'] != "CS") & (df_graph['reqattr'] != "N/A")]
    requests_received = df_graph[(df_graph['dst'] == node) & (df_graph['reqattr'].notnull()) & (
        df_graph['reqattr'] != "CS") & (df_graph['reqattr'] != "N/A")]
    num_requests_sent = len(requests_sent)
    num_requests_received = len(requests_received)

    # Request flow features
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    predecessors_type = [G.nodes[x].get('type') for x in predecessors]
    num_script_predecessors = len(
        [x for x in predecessors_type if x == "Script"])
    successors_type = [G.nodes[x].get('type') for x in successors]
    num_script_successors = len([x for x in successors_type if x == "Script"])

    rf_features = [num_script_predecessors, num_script_successors, num_requests_sent,
                   num_requests_received]

    rf_feature_names = ['num_script_predecessors', 'num_script_successors', 'num_requests_sent',
                        'num_requests_received']

    return rf_features, rf_feature_names


def get_indirect_features(G, df_graph, node):
    """
    Function to extract indirect edge features.

    Args:
      G: networkX graph of indirect edges
      df_graph: DataFrame representation of graph (indirect edges only)
      node: URL of node
    Returns:
      indirect_features: indirect feature values
      indirect_feature_names: indirect feature names
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1
    num_exfil = 0
    num_url_exfil = 0
    num_header_exfil = 0
    num_body_exfil = 0
    num_cookieheader_exfil = 0
    num_infil = 0
    num_infil_content = 0
    num_ls_exfil = 0
    num_ls_url_exfil = 0
    num_ls_header_exfil = 0
    num_ls_body_exfil = 0
    num_ls_cookieheader_exfil = 0
    num_ls_infil = 0
    num_ls_infil_content = 0
    num_ls_src = 0
    num_ls_dst = 0

    try:
        if len(df_graph) > 0:
            num_exfil = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'out')])
            num_infil = len(df_graph[(df_graph['src'] == node)
                            & (df_graph['direction'] == 'in')])
            num_infil_content = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'in') & (df_graph['type'] == 'content')])
            num_url_exfil = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'url')])
            num_header_exfil = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'header')])
            num_body_exfil = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'postbody')])

            node_ls_name = node + "|$$|LS"
            ls_exfil = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'out')])
            ls_infil = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'in')])
            ls_infil_content = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'in') & (df_graph['type'] == 'content')])
            ls_url_exfil = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'url')])
            ls_header_exfil = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'header')])
            ls_body_exfil = len(df_graph[(df_graph['src'] == node_ls_name) & (
                df_graph['direction'] == 'out') & (df_graph['type'] == 'postbody')])

            num_ls_exfil = num_exfil + ls_exfil
            num_ls_infil = num_infil + ls_infil
            num_ls_infil_content = num_infil_content + ls_infil_content
            num_ls_url_exfil = num_url_exfil + ls_url_exfil
            num_ls_header_exfil = num_header_exfil + ls_header_exfil
            num_ls_body_exfil = num_body_exfil + ls_body_exfil

            num_ls_src = len(df_graph[(df_graph['src'] == node) & (
                df_graph['direction'] == 'local')])
            num_ls_dst = len(df_graph[(df_graph['dst'] == node) & (
                df_graph['direction'] == 'local')])

        if (len(G.nodes()) > 0) and node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [
                *nx.average_degree_connectivity(G, nodes=[node]).values()][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1

    except Exception as e:
        traceback.print_exc()
        print(e)

    indirect_features = [in_degree, out_degree, ancestors, descendants, closeness_centrality,
                         average_degree_connectivity, eccentricity, num_exfil, num_infil, num_infil_content, num_url_exfil,
                         num_header_exfil, num_body_exfil, num_ls_exfil, num_ls_infil, num_ls_infil_content, num_ls_url_exfil,
                         num_ls_header_exfil, num_ls_body_exfil,
                         num_ls_src, num_ls_dst]

    indirect_feature_names = ['indirect_in_degree', 'indirect_out_degree', 'indirect_ancestors', 'indirect_descendants', 'indirect_closeness_centrality',
                              'indirect_average_degree_connectivity', 'indirect_eccentricity', 'num_exfil', 'num_infil', 'num_infil_content',
                              'num_url_exfil', 'num_header_exfil', 'num_body_exfil', 'num_ls_exfil', 'num_ls_infil', 'num_ls_infil_content', 'num_ls_url_exfil',
                              'num_ls_header_exfil', 'num_ls_body_exfil',
                              'num_ls_src', 'num_ls_dst']

    return indirect_features, indirect_feature_names


def get_indirect_all_features(G, node):
    """
    Function to extract dataflow features of graph with both direct and indirect edges ('indirect_all').

    Args:
      G: networkX graph (of both direct and indirect edges)
      node: URL of node
    Returns:
      indirect_all_features: indirect_all feature values
      indirect_all_feature_names: indirect_all feature names
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1

    try:
        if (len(G.nodes()) > 0) and (node in G.nodes()):
            in_degree = G.in_degree(node)
            out_degree = G.in_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [
                *nx.average_degree_connectivity(G, nodes=[node]).values()][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1
    except Exception as e:
        LOGGER.warning(
            "[ get_indirect_all_features ] : ERROR - ", exc_info=True)

    indirect_all_features = [in_degree, out_degree, ancestors, descendants,
                             closeness_centrality, average_degree_connectivity, eccentricity]
    indirect_all_feature_names = ['indirect_all_in_degree', 'indirect_all_out_degree',
                                  'indirect_all_ancestors', 'indirect_all_descendants',
                                  'indirect_all_closeness_centrality',
                                  'indirect_all_average_degree_connectivity',
                                  'indirect_all_eccentricity']

    return indirect_all_features, indirect_all_feature_names


def get_dataflow_features(G, df_graph, node, dict_redirect, G_indirect, G_indirect_all, df_indirect_graph):
    """
    Function to extract dataflow features. This function calls
    the other functions to extract different types of dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
      node: URL of node
      dict_redirect: dictionary of redirect depths for each node
      G_indrect: networkX graph of indirect edges
      G_indirect_all: networkX graph of direct and indirect edges
      df_indirect_graph: DataFrame representation of indirect edges
    Returns:
      all_features: dataflow feature values
      all_feature_names: dataflow feature names
    """

    all_features = []
    all_feature_names = []

    setter_domain = df_graph[(df_graph['graph_attr'] == 'Node') & (
        df_graph['name'] == node)]['setter_domain'].iloc[0]

    storage_features, storage_feature_names = get_storage_features(
        df_graph, node)
    # redirect_features, redirect_feature_names = get_redirect_features(df_graph, node, dict_redirect) # we do not need redirect features in cookiegraph
    rf_features, rf_feature_names = get_request_flow_features(
        G, df_graph, node)
    indirect_features, indirect_feature_names = get_indirect_features(
        G_indirect, df_indirect_graph, node)
    indirect_all_features, indirect_all_feature_names = get_indirect_all_features(
        G_indirect_all, node)

    setter_features, setter_feature_names = get_setter_features(
        df_graph, df_indirect_graph, node, setter_domain)

    all_features = storage_features + rf_features + \
        indirect_features + indirect_all_features + setter_features
    all_feature_names = storage_feature_names + rf_feature_names + \
        indirect_feature_names + indirect_all_feature_names + setter_feature_names

    return all_features, all_feature_names


def pre_extraction(G, df_graph, ldb):
    """
    Function to obtain indirect edges before calculating dataflow features.

    Args:
      G: networkX graph
      df_graph: DataFrame representation of graph
    Returns:
      dict_redirect: dictionary of redirect depths for each node
      G_indrect: networkX graph of indirect edges
      G_indirect_all: networkX graph of direct and indirect edges
      df_indirect_edges: DataFrame representation of indirect edges
    """

    G_indirect = nx.DiGraph()

    df_exfiltration = find_exfiltrations(df_graph, ldb)
    # dict_redirect = get_redirect_depths(df_graph) # we are not using this in cookiegraph so removing for now
    df_indirect_edges = df_exfiltration.reset_index()

    if len(df_indirect_edges) > 0:
        G_indirect = nx.from_pandas_edgelist(
            df_indirect_edges, source='src', target='dst', edge_attr=True, create_using=nx.DiGraph())
    G_indirect_all = nx.compose(G, G_indirect)

    return G_indirect, G_indirect_all, df_indirect_edges
