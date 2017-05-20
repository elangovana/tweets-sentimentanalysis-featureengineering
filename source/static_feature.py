import logging
import os
from operator import contains

import arff

import pandas as pd
from pandas import  np

from text_analyser import TextAnalyser


class StaticFeature:
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

        #get word presence/against
        self.logger.info("Get word presence / absence against the tweet")
        vcontains = np.vectorize(contains)
        for w in self.features:
            df_tweet[w] = vcontains(df_tweet["words"], w)

        #Start formatting for arff format
        self.logger.info("Start formatting for arff")
        df_arff = pd.merge(df_tweet, df_label, left_index=True, right_index=True)
        df_arff["id"] = list(df_arff.index)

        attributes = []#[("id", 'NUMERIC')]
        attributes = attributes + [(x, ['True', 'False']) for x in self.features]
        attributes = attributes + [("sentiment", ['positive', 'negative', 'neutral'])]

        obj = {
            'description': u'',
            'relation': 'twitter',
            'attributes': attributes,
            'data': df_arff[self.features + ["sentiment"]].values.tolist(),
        }

        with open(os.path.join(self.output_dir, output_file_name), 'a') as ofile:
            arff.dump(obj, ofile)

        self.logger.info("--Completed----")