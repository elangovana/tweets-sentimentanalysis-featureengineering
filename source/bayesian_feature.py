import logging
import os
from operator import contains

import arff

import pandas as pd
from pandas import DataFrame, np

from text_analyser import TextAnalyser


class BayesianFeature:
    def __init__(self, output_dir, logger=None):
        self.output_dir = output_dir
        self.logger = logger
        self.logger = logger or logging.getLogger(__name__)

    def LogSummary(self):
        self.logger.info("--Summary--")

        self.logger.info("--End of summary--")

    def get_features(self, df_tweet, df_label):
        self.features = ["amazing", "antman", "awesome", "best", "birthday", "cream", "day", "death", "drone",
                         "excited", "fake",
                         "fuck", "fucking", "good", "great", "gucci", "happy", "hate", "ice", "leftists", "liberals",
                         "love", "national",
                         "nazi", "night", "obama", "people", "racist", "see", "shit", "stupid", "supremacists",
                         "tomorrow", "trump"]

        print (df_tweet)

    def generate_arff(self, df_tweet, df_label, output_file_name):
        text_analyser = TextAnalyser()

        vget_words = np.vectorize(lambda x: set(text_analyser.get_words_without_stopwords(x)))
        # get non-stop words
        df_tweet["words"] = vget_words(df_tweet["tweet"])

        vcontains = np.vectorize(contains)
        df_arff = pd.merge(df_tweet, df_label, left_index=True, right_index=True)
        df_arff["id"] = list(df_arff.index)
        for w in self.features:
            df_arff[w] = vcontains(df_tweet["words"], w)

        attributes = [("id", 'NUMERIC')]
        attributes = attributes + [(x, ['True', 'False']) for x in self.features]
        attributes = attributes + [("sentiment", ['positive', 'negative', 'neutral'])]

        obj = {
            'description': u'',
            'relation': 'twitter',
            'attributes': attributes,
            'data': df_arff[["id"]+self.features + ["sentiment"]].values.tolist(),
        }
        print ("obj")
        print (obj)
        with open(os.path.join(self.output_dir, output_file_name), 'a') as ofile:
            arff.dump(obj, ofile)
