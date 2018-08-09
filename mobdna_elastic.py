# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

ELASTICSEARCH FUNCTIONS

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import csv
import json
import sys
from pprint import PrettyPrinter

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

import mobdna_helper as help

# Dashboard
pp = PrettyPrinter(indent=4)
fieldnames = {'appevents':
                  ['application',
                   'battery',
                   'data_version',
                   'endTime',
                   'endTimeMillis',
                   'id',
                   'latitude',
                   'longitude',
                   'model',
                   'notification',
                   'session',
                   'startTime',
                   'startTimeMillis'],
              'notifications':
                  ['id',
                   'notificationID',
                   'application',
                   'time',
                   'posted',
                   'data_version'],
              'sessions':
                  ['id',
                   'timestamp',
                   'session on',
                   'data_version'],
              'logs':
                  ['id',
                   'date',
                   'logging enabled']}


def connect(server='172.18.120.104', port=9200) -> Elasticsearch:
    """Establish connection with data"""

    es = Elasticsearch(
        hosts=[{'host': server, 'port': port}],
        timeout=60,
        max_retries=10,
        retry_on_timeout=True
    )
    return es


def ids_from_file(dir='', file_name='ids', file_type='csv') -> list:
    """Read ids from file."""

    # Create path
    path = dir + ('/' if dir != '' else '') + file_name + '.' + file_type

    # Initialize id list
    id_list = []

    # Open file and read lines
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            id_list.append(row[0])

    return id_list


def ids_from_server(based_on = "appevents") -> dict:
    """Fetch ids from server. Returns count of appevents."""

    # Check argument
    if based_on not in {"appevents", "sessions", "notifications"}:

        raise Exception("Must be based on appevents, sessions or notifications!")

    es = connect()

    # ID query
    body = {
        "size": 0,
        "aggs": {
            "unique_id": {
                "terms": {
                    "field": "id.keyword",
                    "size": 1000000
                }
            }
        }
    }

    # Search using scroller (avoid overload)
    res = es.search(index="mobiledna",
                    body=body,
                    request_timeout=300,
                    #scroll='30s',  # Get scroll id to get next results
                    doc_type=based_on)

    # Get the ids
    ids = {}

    # Go over buckets and get count
    for b in res['aggregations']['unique_id']['buckets']:

        ids[b['key']] = b['doc_count']

    return ids


def fetch(doc_type: str, ids: list, time_range=('2017-01-01T00:00:00.000', '2020-01-01T00:00:00.000')) -> dict:
    """Fetch data from server"""

    # Are we looking for the right doc_types?
    if doc_type not in {"appevents", "notifications", "sessions", "logs"}:

        raise Exception("Can't fetch data for anything other than appevents,"
                        " notifications or sessions (or logs, but whatever).")

    # If there's more than one id, recursively call this function
    if len(ids) > 1:

        # Save all results in dict, with id as key
        dump_dict = {}

        # Go over ids and try to fetch data
        for index, id in enumerate(ids):

            print("ID {index}: \t{id}".format(index=index + 1, id=id))
            try:
                dump_dict[id] = fetch(doc_type=doc_type, ids=[id], time_range=time_range)[id]
            except:
                print("Fetch failed for {id}".format(id=id))

        return dump_dict

    # If there's one id, fetch data
    else:

        # Establish connection
        es = connect()

        # Create query
        # - Depending on doc_type, a different variable is used
        # - to restrict the time range (beautiful coding by ItP :/)
        time_var = {
            'appevents': 'startTime',
            'notifications': 'time',
            'sessions': 'timestamp',
            'logs': 'date'
        }

        # Base query
        body = {
            'query': {
                'constant_score': {
                    'filter': {
                        'bool': {
                            'must': [
                                {
                                    'terms':
                                        {'id.keyword':
                                             ids
                                         }
                                }
                            ]

                        }
                    }
                }
            }
        }

        # Chance query if time is factor
        try:
            start = time_range[0]
            stop = time_range[1]
            range_restriction = {
                'range':
                    {time_var[doc_type]:
                         {'format': "yyyy-MM-dd'T'HH:mm:ss.SSS",
                          'gte': start,
                          'lte': stop}
                     }
            }
            body['query']['constant_score']['filter']['bool']['must'].append(range_restriction)

        except:
            print("Failed to restrict range. Getting all data.")

        # Count entries
        count_tot = es.count(index="mobiledna", doc_type=doc_type)
        count_ids = es.count(index="mobiledna", doc_type=doc_type, body=body)

        print("There are {count} entries of the type <{doc_type}>.".format(count=count_tot["count"], doc_type=doc_type))
        print("Selecting {ids} leaves {count} entries.".format(ids=ids, count=count_ids["count"]))

        # Search using scroller (avoid overload)
        res = es.search(index="mobiledna",
                        body=body,
                        request_timeout=60,
                        size=1000,  # Get first 1000 results
                        scroll='30s',  # Get scroll id to get next results
                        doc_type=doc_type)

        # Update scroll id
        scroll_id = res['_scroll_id']
        total_size = res['hits']['total']

        # Save all results in list
        dump = []

        # Get data
        temp_size = total_size

        ct = 0
        while 0 < temp_size:
            ct += 1
            res = es.scroll(scroll_id=scroll_id,
                            scroll='30s',
                            request_timeout=60)
            dump += res['hits']['hits']
            scroll_id = res['_scroll_id']
            temp_size = len(res['hits']['hits'])  # As long as there are results, keep going ...
            remaining = (total_size - (ct * 1000)) if (total_size - (ct * 1000)) > 0 else temp_size
            sys.stdout.write("Entries remaining: {rmn}\r".format(rmn=remaining))
            sys.stdout.flush()

        es.clear_scroll(body={'scroll_id': [scroll_id]})  # Cleanup (otherwise Scroll id remains in ES memory)

        return {ids[0]: dump}


def export_elastic(dir: str, name: str, doc_type: str, data: dict, pickle=True, csv_file=False):
    """Export data to file type (standard pickle, csv possible)."""

    # Did we get data?
    if data is None:
        raise Exception("Received empty data. Failed to export.")

    # Gather data to write to CSV
    to_export = []
    for d in data.values():
        for dd in d:
            to_export.append(dd['_source'])

    # Create path
    path = help.set_dir(dir)

    # Export file to pickle
    if pickle:
        df = pd.DataFrame(to_export)
        df.to_pickle(path + name + '_' + doc_type + '.pkl')

    # Export file to csv
    if csv_file:
        with open(path + name + '_' + doc_type + '.csv', 'w') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames[doc_type], delimiter=';')
            writer.writeheader()
            for r in to_export:
                writer.writerow(r)


def pipeline(dir: str, name: str, ids: list,
             time_range=('2017-12-06T00:00:00.000', '2020-01-01T00:00:00.000'),
             pickle=True, csv_file=False):
    """Get all doc_types sequentially."""

    # All data
    all_df = {}

    # Go over interesting doc_types
    for doc_type in {"appevents", "notifications", "sessions"}:
        # Get data from server
        print("\nGetting " + doc_type + "...\n")
        data = fetch(doc_type=doc_type, ids=ids, time_range=time_range)

        # Export data
        print("\n\nExporting " + doc_type + "...")
        export_elastic(dir=dir, name=name, doc_type=doc_type, data=data, csv_file=csv_file, pickle=pickle)

        all_df[doc_type] = data

    print("\nALL DONE!")

    return all_df


if __name__ == "__main__":

    # Sup?
    help.hi()

    '''
    # What are we looking for:
    try:
        project = sys.argv[1]
    except:
        project = "xavier"

    # Read ids we're interested in
    ids = ids_from_file(file_name='ids')
    pp.pprint(ids)
    # Get data for these parameters (NOTE: can add time restricting by adding time_range tuple)
    #data = fetch(doc_type=doc_type, ids=ids)
    # export(dir="data", name='francine', file_type='csv', doc_type=doc_type, data=data)
    # df = to_df(data=data)
    data = pipeline(dir="data/" + project, name=project, ids=ids, pickle=True)
    '''

    ids = ids_from_server(based_on="notifications")
    np.save("ids.npy", ids)

    with open('ids.json', 'w') as fp:
        json.dump(ids, fp=fp)

