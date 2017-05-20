import os
from setup_logger import setup_log
import pandas as pd


def run(traindata_tweet_file, traindata_label_file, output_dir, samplesize=2000 ):
    traindata_tweet_file = os.path.join(os.path.dirname(__file__), traindata_tweet_file)
    traindata_label_file = os.path.join(os.path.dirname(__file__), traindata_label_file)

    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    os.makedirs(output_dir)
    logger = setup_log(output_dir)

    df_traindata_tweet = pd.read_csv(traindata_tweet_file, sep='\t', header=None, names=["id", "tweet"], dtype=object)
    df_traindata_label = pd.read_csv(traindata_label_file, sep='\t', header=None, names=["id", "sentiment"], keep_default_na=False)

    if (samplesize > 0):
        df_traindata_tweet = df_traindata_tweet.sample(samplesize)



if __name__ == "__main__":
   run("../inputdata/train-tweets.txt", "../inputdata/train-labels.txt", "../outputdata")