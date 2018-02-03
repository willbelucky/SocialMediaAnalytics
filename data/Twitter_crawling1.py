import csv
import json
from collections import defaultdict

import pandas as pd
import tweepy

DATA_DIR = 'data/'

consumer_key = 'V5KLfqh2ulO5ySisRXzuPoJlN'
consumer_secret = 'Fonp2HFZaqsegbHbz1WFhzDsQoaiz5FhRcsh6s0rqYp2fqWhhl'
access_token = '956673073943588865-YJdtTdDFZsOW149FHL0BwDxdetxHTn3'
access_token_secret = 'is2FXrK4MFRFDW9VzMvgAPfjKWBxNjB1h12WlQn7BeltG'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)
query = 'sport'
max_tweets = 5000
searched_tweets = [status._json for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]
json_strings = [json.dumps(json_obj) for json_obj in searched_tweets]

d = defaultdict(list)
"""
    A (i)   Tweets   A               (A-A Tweets)                   
      (ii)  RT       B  Mentions C   (A-B RT, B-C Mentions)
                        Tweets   B   (A-B RT, B-B Tweets)
      (iii) Mentions C               (A-C Mentions)

"""

for i in range(max_tweets):
    datastore = json.loads(json_strings[i])
    # A-A Tweets
    if len(datastore["entities"]["user_mentions"]) == 0:

        # column1.
        print(datastore["user"]["screen_name"])
        d["Column1"].append(str("@" + datastore["user"]["screen_name"]))

        # column2.
        print(datastore["user"]["screen_name"])
        d["Column2"].append(str("@" + datastore["user"]["screen_name"]))

        # column3.
        print("tweet")
        d["Column3"].append("tweet")

    else:
        # A-B RT
        number_of_users = len(datastore["entities"]["user_mentions"])
        last_person_index = datastore["entities"]["user_mentions"][number_of_users - 1]["indices"]

        if last_person_index[0] == 3:

            # column1
            print(datastore["user"]["screen_name"])
            d["Column1"].append(str("@" + datastore["user"]["screen_name"]))

            # column2
            print(datastore["entities"]["user_mentions"][0]['screen_name'])
            d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][0]['screen_name']))

            # column3
            print("RT")
            d["Column3"].append("RT")

            # B-B tweet
            print(datastore["entities"]["user_mentions"][0]['screen_name'])
            d["Column1"].append(str("@" + datastore["entities"]["user_mentions"][0]['screen_name']))

            print(datastore["entities"]["user_mentions"][0]['screen_name'])
            d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][0]['screen_name']))

            print("tweet")
            d["Column3"].append("tweet")

        # A-C Mentions
        elif last_person_index[0] != 3 and datastore["text"][0:4] != "RT @":
            for k in range(len(datastore["entities"]["user_mentions"])):
                # column1
                print(datastore["user"]["screen_name"])
                d["Column1"].append(str("@" + datastore["user"]["screen_name"]))

                # column2
                print(datastore["entities"]["user_mentions"][k]['screen_name'])
                d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][k]['screen_name']))

                # column3
                print("Mentions")
                d["Column3"].append("Mentions")

        else:
            # more than 2 people, start from RT @
            for j in range(number_of_users):
                index_j = datastore["entities"]["user_mentions"][j]["indices"][0]
                if datastore["text"][(index_j - 3):(index_j - 1)] == "RT":

                    # column1
                    print(datastore["user"]["screen_name"])
                    d["Column1"].append(str("@" + datastore["user"]["screen_name"]))

                    # column2
                    print(datastore["entities"]["user_mentions"][j]['screen_name'])
                    d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][j]['screen_name']))

                    # column3
                    print("RT")
                    d["Column3"].append("RT")
                    last_RT = j

                    if last_RT == number_of_users - 1:
                        print(datastore["entities"]["user_mentions"][last_RT]['screen_name'])
                        d["Column1"].append(str("@" + datastore["entities"]["user_mentions"][last_RT]['screen_name']))

                        print(datastore["entities"]["user_mentions"][last_RT]['screen_name'])
                        d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][last_RT]['screen_name']))

                        print(datastore["entities"]["user_mentions"][last_RT]['screen_name'])
                        d["Column3"].append("tweets")
                    else:
                        continue
                else:

                    # B-C mentions
                    print(datastore["entities"]["user_mentions"][last_RT]['screen_name'])
                    d["Column1"].append(str("@" + datastore["entities"]["user_mentions"][last_RT]['screen_name']))

                    # column2
                    print(datastore["entities"]["user_mentions"][j]['screen_name'])
                    d["Column2"].append(str("@" + datastore["entities"]["user_mentions"][j]['screen_name']))

                    # column3
                    print("Mentions")
                    d["Column3"].append("Mentions")

df = pd.DataFrame.from_dict(d)
df.reset_index(drop=True)
df.to_csv(DATA_DIR + 'my_file.csv', index=False)
