import getopt
import os

import sys

import time


import re

from bayesian_feature import BayesianFeature
from setup_logger import setup_log
import pandas as pd
import  numpy as np
from static_feature import StaticFeature
from text_analyser import TextAnalyser


def run(traindata_tweet_file, traindata_label_file, devdata_tweet_file,devdata_label_file,output_dir, samplesize=2000 ):

    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir)
    logger = setup_log(output_dir)

    df_devdata_label, df_devdata_tweet = GetTweetDataFrame(samplesize, devdata_label_file,r"\t", devdata_tweet_file,logger)
    df_traindata_label, df_traindata_tweet = GetTweetDataFrame(samplesize, traindata_label_file, "\t",traindata_tweet_file,logger)

    ##run 1
    logger.info("----Running static features-------")
    resultsdir = os.path.join(output_dir, "Run_static_{}".format(time.strftime('%Y%m%d_%H%M%S')))
    os.makedirs(resultsdir)
    feature_engineer = StaticFeature(resultsdir, logger)
    feature_engineer.get_features(df_traindata_tweet.copy(), df_traindata_label.copy())
    feature_engineer.generate_arff(df_traindata_tweet.copy(), df_traindata_label.copy(),"train.arff")
    feature_engineer.generate_arff(df_devdata_tweet.copy(), df_devdata_label.copy(), "dev.arff")

    ##run 2
    logger.info("----Running bayesiean features-------")
    resultsdir = os.path.join(output_dir, "Run_bayesiean_{}".format(time.strftime('%Y%m%d_%H%M%S')))
    os.makedirs(resultsdir)
    feature_engineer = BayesianFeature(resultsdir, logger)
    feature_engineer.get_features(df_traindata_tweet.copy(), df_traindata_label.copy())
    feature_engineer.generate_arff(df_traindata_tweet.copy(), df_traindata_label.copy(), "train.arff")
    feature_engineer.generate_arff(df_devdata_tweet.copy(), df_devdata_label.copy(), "dev.arff")


def GetTweetDataFrame(samplesize, label_file, delimiter, tweet_file, logger):
    tweet_file = os.path.join(os.path.dirname(__file__), tweet_file)
    label_file = os.path.join(os.path.dirname(__file__), label_file)
    print ("----GetTweetDataFrame---")
    #parse file
    df_tweet = pd.read_csv(tweet_file, delimiter=delimiter,skipinitialspace=True, header=None, names=["id", "tweet"], dtype=object,
                                     index_col=0)
    df_label = pd.read_csv(label_file, delimiter="\t", skipinitialspace=True,header=None, names=["id", "sentiment"],
                                     keep_default_na=False, index_col=0)
    print (df_label.columns)
    print (np.size(df_label))
    print (df_tweet.columns)
    print (np.size(df_tweet))
    if (samplesize > 0):
        df_tweet = df_tweet.sample(samplesize)
        df_label = pd.merge(df_label, df_tweet, left_index=True, right_index=True)

        print (df_label.columns)
        print (np.size(df_label))
        print (df_tweet.columns)
        print (np.size(df_tweet))


   # get non-stop words

    logger.info("Obtaining non-stop words from the tweet")
    # TODO Unicode hack, filter features with printable charcters only


    #stem word with chracters ignore words with ascii
    vget_non_stop_words = np.vectorize(stemwords1, otypes=[set])

    df_tweet["words"]=vget_non_stop_words(df_tweet["tweet"])
    print (df_label.columns)
    print (np.size(df_label))
    print (df_tweet.columns)
    print (np.size(df_tweet))
    return df_label, df_tweet

def stemwords1(sentence):
    text_analyser = TextAnalyser()

    without_stopwords = text_analyser.get_words_without_stopwords(sentence)

    words = set(without_stopwords)

    words_with_letters_only= [w for w in words if re.match('^[a-zA-Z]+$',w) ]

    if len(words_with_letters_only) ==0:
        return set()
    tokens= text_analyser.Stem_tokens(words_with_letters_only)
    result = set()
    tmp= [result.add(x) for x in tokens]
    print result
    return  result




def main(argv):
    tweet_file="../inputdata/train-tweets.txt"
    label_file="../inputdata/train-labels.txt"
    dev_tweet_file="../inputdata/dev-tweets.txt"
    dev_label_file="../inputdata/dev-labels.txt"
    outdir="../outputdata/train_{}".format(time.strftime('%Y%m%d_%H%M%S'))
    samplesize=1000
    try:
        opts, args = getopt.getopt(argv, "ht:l:o:s", ["tweetfile=", "labelfile=","outdir=" ,"samplesize="])
    except getopt.GetoptError:
        print 'main.py -i <tweetfile> -l <labelfile> -o <outputdir>  [-s  <samplesize>]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print '-i <tweetfile> -l <labelfile> -o <outputdir> [ -s  <samplesize>]'
            sys.exit()
        elif opt in ("-t", "--tweetfile"):
            tweet_file = arg
        elif opt in ("-o", "--outdir"):
            outdir = arg
        elif opt in ("-l", "--labelfile"):
            label_file = arg
        elif opt in ("-s", "--samplesize"):
            samplesize = int(arg)
    run(tweet_file, label_file, dev_tweet_file, dev_label_file, outdir, samplesize)

if __name__ == "__main__":
   main(sys.argv[1:])
