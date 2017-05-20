import getopt
import os

import sys

import time

from bayesian_feature import BayesianFeature
from setup_logger import setup_log
import pandas as pd

from static_feature import StaticFeature
from text_analyser import TextAnalyser


def run(traindata_tweet_file, traindata_label_file, devdata_tweet_file,devdata_label_file,output_dir, samplesize=2000 ):
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir)
    logger = setup_log(output_dir)

    df_traindata_label, df_traindata_tweet = GetTweetDataFrame(samplesize, traindata_label_file, traindata_tweet_file,logger)
    df_devdata_label, df_devdata_tweet = GetTweetDataFrame(samplesize, devdata_label_file, devdata_tweet_file,logger)

    ##run 1
    logger.info("----Running static features-------")
    resultsdir = os.path.join(output_dir, "Run_{}".format(time.strftime('%Y%m%d_%H%M%S')))
    os.makedirs(resultsdir)
    feature_engineer = StaticFeature(resultsdir, logger)
    feature_engineer.get_features(df_traindata_tweet.copy(), df_traindata_label.copy())
    feature_engineer.generate_arff(df_traindata_tweet.copy(), df_traindata_label.copy(),"train.arff")
    feature_engineer.generate_arff(df_devdata_tweet.copy(), df_devdata_label.copy(), "dev.arff")

    ##run 2
    logger.info("----Running bayesiean features-------")
    resultsdir = os.path.join(output_dir, "Run_{}".format(time.strftime('%Y%m%d_%H%M%S')))
    os.makedirs(resultsdir)
    feature_engineer = BayesianFeature(resultsdir, logger)
    feature_engineer.get_features(df_traindata_tweet.copy(), df_traindata_label.copy())
    feature_engineer.generate_arff(df_traindata_tweet.copy(), df_traindata_label.copy(), "train.arff")
    feature_engineer.generate_arff(df_devdata_tweet.copy(), df_devdata_label.copy(), "dev.arff")


def GetTweetDataFrame(samplesize, traindata_label_file, traindata_tweet_file, logger):
    traindata_tweet_file = os.path.join(os.path.dirname(__file__), traindata_tweet_file)
    traindata_label_file = os.path.join(os.path.dirname(__file__), traindata_label_file)

    #parse file
    df_traindata_tweet = pd.read_csv(traindata_tweet_file, sep='\t', header=None, names=["id", "tweet"], dtype=object,
                                     index_col=0)
    df_traindata_label = pd.read_csv(traindata_label_file, sep='\t', header=None, names=["id", "sentiment"],
                                     keep_default_na=False, index_col=0)


    if (samplesize > 0):
        df_traindata_tweet = df_traindata_tweet.sample(samplesize)
        df_traindata_label = pd.merge(df_traindata_label, df_traindata_tweet, left_index=True, right_index=True)

   # get non-stop words
    text_analyser = TextAnalyser()
    logger.info("Obtaining non-stop words from the tweet")
    vget_non_stop_words = pd.np.vectorize(lambda x: set(text_analyser.get_words_without_stopwords(x)))
    df_traindata_tweet["words"] = vget_non_stop_words(df_traindata_tweet["tweet"])

    return df_traindata_label, df_traindata_tweet


def main(argv):
    tweet_file="../inputdata/train-tweets.txt"
    label_file="../inputdata/train-labels.txt"
    dev_tweet_file="../inputdata/train-tweets.txt"
    dev_label_file="../inputdata/train-labels.txt"
    outdir="../outputdata/train_{}".format(time.strftime('%Y%m%d_%H%M%S'))
    samplesize=500
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