import pickle
import pandas as pd

from parsers import config

top_n = 50

dataPath = "./data/messenger.pkl"

df = pd.DataFrame()
df = pd.concat([df, pd.read_pickle(dataPath)])
df.columns = config.ALL_COLUMNS

print ("Loaded {} messages.".format(len(df)))

mf = df.groupby(['conversationWithName'], as_index=False) \
        .agg(lambda x: len(x)) \
        .sort_values('timestamp', ascending=False) \
        .head(top_n)


print(mf)
