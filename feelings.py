#!/usr/bin/env python3
import argparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from ggplot import *

df = pd.DataFrame()
df = pd.concat([df, pd.read_pickle('data/messenger.pkl')])

df.columns = ['timestamp', 'conversationId', 'conversationWithName', 'senderName', 'text', 'language', 'platform', 'datetime']
print('Loaded', len(df), 'messages')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--own-name', dest='own_name', type=str,
                        help='name of the owner of the chat logs, written as in the logs', required=True)
    parser.add_argument('--friend-name', '--file-path', dest='friend_name', help='Friend name', required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()

#=df3 = df3[df.text.notnull()].dropna()
df3 = df[df.language == 'en']
df3 = df3[df3.text.map(len) > 10]

sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
friend = df[df.conversationWithName == args.friend_name]

print(args.friend_name)
print(len(friend))

print(friend)

mf = friend.sort_values('sentiment', ascending=True)
print(mf)


print(ggplot(aes(x='sentiment', fill='senderName'), data=friend) + geom_density(alpha=0.7))
