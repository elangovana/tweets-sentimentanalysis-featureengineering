# tweets-sentimentanalysis-featureengineering
This Python scripts creates the top most commonly occuring word presence as features 

### Prerequisties 
1. Install Python 2.7 from https://www.python.org/downloads/release/python-2713/
2. Create a virtual environment as detailed in http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/.
3. Within the virtual environment Install python package cython using the command
    pip install cython

### Set up this python module.
In your virtual environment run 
> python setup.py develop

### How to run 
To get help
> python main.py -h

To run this script using the command  and use only 100 records from the training data/dev & test set, with output in outdir use
> python main.py  -t ../inputdata/train-tweets.txt  -l ../inputdata/train-labels.txt -d ../inputdata/dev-tweets.txt  -v ../inputdata/dev-labels.txt  -x ../inputdata/test-tweets.txt -o outdir  --samplesize 100
  
