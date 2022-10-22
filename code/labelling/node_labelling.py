import json
import pandas as pd
from code.graph.database import Database
import tldextract


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
        return None


def get_resource_type(attr):
    """
    Function to get resource type of a node.

    Args:
        attr: Node attributes.
    Returns:
        Resource type of node.
    """

    try:
        attr = json.loads(attr)
        return attr['content_policy_type']
    except Exception as e:
        return None


def match_url(domain_top_level, current_domain, current_url, resource_type, rules_dict):
    """
    Function to match node information with filter list rules.

    Args:
        domain_top_level: eTLD+1 of visited page.
        current_domain; Domain of request being labelled.
        current_url: URL of request being labelled.
        resource_type: Type of request being labelled (from content policy type).
        rules_dict: Dictionary of filter list rules.
    Returns:
        Label indicating whether the rule should block the node (True/False).
    """

    try:
        if domain_top_level == current_domain:
            third_party_check = False
        else:
            third_party_check = True

        if resource_type == 'sub_frame':
            subdocument_check = True
        else:
            subdocument_check = False

        if resource_type == 'script':
            if third_party_check:
                rules = rules_dict['script_third']
                options = {'third-party': True, 'script': True,
                           'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['script']
                options = {'script': True, 'domain': domain_top_level,
                           'subdocument': subdocument_check}

        elif resource_type == 'image' or resource_type == 'imageset':
            if third_party_check:
                rules = rules_dict['image_third']
                options = {'third-party': True, 'image': True,
                           'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['image']
                options = {'image': True, 'domain': domain_top_level,
                           'subdocument': subdocument_check}

        elif resource_type == 'stylesheet':
            if third_party_check:
                rules = rules_dict['css_third']
                options = {'third-party': True, 'stylesheet': True,
                           'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['css']
                options = {'stylesheet': True, 'domain': domain_top_level,
                           'subdocument': subdocument_check}

        elif resource_type == 'xmlhttprequest':
            if third_party_check:
                rules = rules_dict['xmlhttp_third']
                options = {'third-party': True, 'xmlhttprequest': True,
                           'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['xmlhttp']
                options = {'xmlhttprequest': True, 'domain': domain_top_level,
                           'subdocument': subdocument_check}

        elif third_party_check:
            rules = rules_dict['third']
            options = {'third-party': True, 'domain': domain_top_level,
                       'subdocument': subdocument_check}

        else:
            rules = rules_dict['domain']
            options = {'domain': domain_top_level,
                       'subdocument': subdocument_check}

        return rules.should_block(current_url, options)

    except Exception as e:
        return False


def get_final_label(row):

    cp_label = row['cat_id_cookiepedia']
    tranco_label = row['cat_id_tranco']

    if pd.isna(cp_label) and pd.isna(tranco_label):
        return pd.NA

    if pd.isna(cp_label):
        return tranco_label
    if pd.isna(tranco_label):
        return cp_label
    return cp_label if cp_label > tranco_label else tranco_label


def clean_up_final_label(label):

    if label < 0:
        return pd.NA
    if label > 3:
        return pd.NA
    return label


def label_node_data(row, filterlists, filterlist_rules):
    """
    Function to label a node with filter lists.

    Args:
        row: Row of node DataFrame.
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    Returns:
        data_label: Label for node (True/False).
    """

    try:
        top_domain = row['top_level_domain']
        url = row['name']
        domain = row['domain']
        resource_type = row['resource_type']
        data_label = False

        for fl in filterlists:
            if top_domain and domain:
                list_label = match_url(
                    top_domain, domain, url, resource_type, filterlist_rules[fl])
                data_label = data_label | list_label
            else:
                data_label = "Error"
    except Exception:
        LOGGER.warning('Error in node labelling:', exc_info=True)
        data_label = "Error"

    return data_label


def label_nodes(df, filterlists, filterlist_rules):
    """
    Function to label nodes with filter lists.

    Args:
        df: DataFrame of nodes.
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    Returns:
        df_nodes: DataFrame of labelled nodes.
    """

    df_nodes = df[(df['type'] != 'Storage') & (df['type'] != 'Element')].copy()
    df_nodes['resource_type'] = df_nodes['attr'].apply(get_resource_type)
    df_nodes['label'] = df_nodes.apply(
        label_node_data, filterlists=filterlists, filterlist_rules=filterlist_rules, axis=1)
    df_nodes = df_nodes[['visit_id', 'name', 'top_level_url', 'label']]

    return df_nodes


def get_categories():

    df_cookiepedia = pd.read_csv(
        "/Users/siby/Documents/webgraph_optimized/labelling_scripts/cookiepedia.csv", index_col=0)
    df_tranco = pd.read_csv(
        "/Users/siby/Documents/webgraph_optimized/labelling_scripts/tranco.csv", index_col=0)

    #df_cookiepedia = pd.read_csv("/home/siby/webgraph_optimized/labelling_scripts/cookiepedia.csv", index_col=0)
    #df_tranco = pd.read_csv("/home/siby/webgraph_optimized/labelling_scripts/tranco.csv", index_col=0)

    df_cookiepedia = df_cookiepedia[['name', 'domain', 'cat_id']]
    df_tranco = df_tranco[['name', 'domain', 'cat_id']]

    df_cookiepedia.drop_duplicates(inplace=True)
    df_tranco.drop_duplicates(inplace=True)

    return df_cookiepedia, df_tranco


def process_declared_labels():

    df_cookiepedia, df_tranco = get_categories()
    df_labels = pd.merge(df_tranco, df_cookiepedia, how='outer', on=[
                         'name', 'domain'], suffixes=('_cookiepedia', '_tranco'))
    df_labels['declared_label'] = df_labels.progress_apply(
        get_final_label, axis=1)
    df_labels.drop_duplicates(inplace=True)
    df_labels = df_labels['declared_label'].groupby(
        [df_labels.name, df_labels.domain]).apply(list).reset_index()
    df_labels['declared_label'] = df_labels.progress_apply(
        lambda x: max(x['declared_label']), axis=1)
    df_labels['domain'] = df_labels.domain.progress_apply(get_domain)
    df_labels['declared_label'] = df_labels.declared_label.progress_apply(
        clean_up_final_label)

    return df_labels


def label_setter_data(row, filterlists, filterlist_rules):

    try:
        top_domain = row['top_level_domain']
        setter_url = row['setter']
        setter_domain = row['setter_domain']
        resource_type = row['resource_type']
        data_label = False

        for fl in filterlists:
            if top_domain and setter_domain:
                list_label = match_url(
                    top_domain, setter_domain, setter_url, resource_type, filterlist_rules[fl])
                data_label = data_label | list_label
            else:
                data_label = "Error"
    except:
        data_label = "Error"

    return data_label


def label_storage_setters(df, filterlists, filterlist_rules):

    df_storage = df[df['type'] == 'Storage']
    df_storage = df_storage[['visit_id', 'name',
                             'setter', 'top_level_domain', 'setter_domain']]
    df_others = df[df['type'] != 'Storage'].copy()
    df_others['resource_type'] = df_others['attr'].apply(get_resource_type)
    df_others = df_others[df_others['resource_type'].notna()]
    df_others = df_others[['name', 'resource_type']]
    df_others = df_others.rename(columns={'name': 'setter'})
    df_merged = df_storage.merge(df_others, on='setter')
    df_merged = df_merged.drop_duplicates()
    df_merged['setter_label'] = df_merged.apply(
        label_setter_data, filterlists=filterlists, filterlist_rules=filterlist_rules, axis=1)

    return df_merged


def get_combined_label(row):

    setter_label = row['setter_label']
    declared_label = row['declared_label']

    try:
        if (pd.notna(declared_label)) and (declared_label == 3):
            return 'Positive'
        elif setter_label == False:
            return 'Negative'
    except:
        return 'Unknown'
    return 'Unknown'


def label_storage_nodes(df_setters, df_declared):

    try:
        df_combined_data = pd.merge(df_setters, df_declared, on=[
                                    'visit_id', 'name'], how='outer')
        df_combined_data['label'] = df_combined_data.apply(
            get_combined_label, axis=1)
        df_combined_data = df_combined_data.drop_duplicates()
    except:
        return pd.DataFrame()

    return df_combined_data


def label_cmp_data(category):

    if category >= 2:
        return True
    else:
        return False


# def label_storage_cookiepedia(df, top_level_domain):

#     df_storage = df[df['type'] == 'Storage']
#     # df_cookiepedia = process_declared_labels()
#     df_labels = pd.read_csv('labels.csv', index_col=0)
#     df_labels = df_labels[df_labels['domain'] == top_level_domain]
#     df_merged = df_storage.merge(
#         df_labels, on = 'name', how='outer')
#     df_merged = df_merged[['visit_id', 'name', 'category']]
#     df_merged = df_merged.drop_duplicates()
#     df_merged['cookiepedia_label'] = df_merged['category'].apply(
#         label_cookiepedia_data)

#     return df_merged



def label_data(df, filterlists, filterlist_rules, top_level_url):

    # df_labelled = label_storage_cookiepedia(df)
    # df_setters = label_storage_setters(df, filterlists, filterlist_rules)
    # df_combined = label_storage_nodes(df_setters, df_labelled)
    # df_labelled = pd.DataFrame()

    # labelling will be handled by a different pre-processing script

    df_labels = pd.read_csv('cmp_labels.csv', index_col=0)
    top_level_domain = get_domain(top_level_url)

    df_labels = df_labels[df_labels['site'] == top_level_domain]

    try:
        df_setters = label_storage_setters(df, filterlists, filterlist_rules)
        df_nodes = df_setters[df['graph_attr'] == "Node"]
        df_nodes = df_nodes[df_nodes['type'] == 'Storage']
        df_labels = df_labels[df_labels['domain'] == top_level_domain]
        df_nodes  = df_nodes.merge(df_labels, on=['name'])
        df_nodes['label'] = df_nodes['category'].apply(label_cmp_data)
        df_nodes.rename(columns={'label': 'declared_label'}, inplace=True)

    except Exception as e:
        LOGGER.warning("Error labelling:", exc_info=True)
        df_nodes = pd.DataFrame()

    return df_nodes
