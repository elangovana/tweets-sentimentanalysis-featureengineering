import csv
import string

import nltk
import nltk.stem.snowball
import nltk.tokenize.punkt
from nltk.tokenize import WordPunctTokenizer
from numpy import sort


class TextAnalyser:
    """
    Analyses text using NLTK toolkit
    """
    def  __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.tokenizer = WordPunctTokenizer()


    def get_words(self, text):
        words = nltk.word_tokenize(text)
        return words

    def get_words_without_punctuation(self, text):
        tokenizer = WordPunctTokenizer()
        tokens = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(text)]
        return tokens


    def get_words_without_stopwords(self, text):
        stopwords = self.stopwords
        stopwords.extend(string.punctuation)
        stopwords.append('')

        tokens = [token.lower().strip(string.punctuation) for token in self.tokenizer.tokenize(text) \
                  if token.lower().strip(string.punctuation) not in stopwords]
        return tokens

    def get_np_chunks(self, text):
        words = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(words)
        chunkgram = r"""Chunk: {(<JJ>*<NN.*>)+}"""
        chunk_parser = nltk.RegexpParser(chunkgram)
        chunked = chunk_parser.parse(tagged)
        return [l for l in self._get_leaves(chunked, "Chunk")]

    def get_nn_chunks(self, text):
        words = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(words)
        chunkgram = r"""Chunk: {(<NN.*>)+}"""
        chunk_parser = nltk.RegexpParser(chunkgram)
        chunked = chunk_parser.parse(tagged)
        return [l for l in self._get_leaves(chunked, "Chunk")]


    def _get_leaves(self, tree, node_type):
        for subtree in tree.subtrees(filter = lambda t: t.label() ==node_type):
            term = [w for (w,t) in subtree.leaves()]
            yield term


    def get_terms(self,leaves):
        for leaf in leaves:
            term = [ w for w,t in leaf]
            yield term


    def get_most_common_words(self, words, top_n_most_common, min_frequency=10):



        fdist = nltk.FreqDist(words)
        print ([ (k,v) for k,v in fdist.most_common(top_n_most_common) ])
        return [ k for k,v in fdist.most_common(top_n_most_common) if v >= min_frequency]



    @staticmethod
    def _write_tuple_to_file(list_of_tuples, file):
        with open(file, 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['name', 'num'])
            for row in list_of_tuples:
                csv_out.writerow(row)

    def aristo_get_most_common_nouns(self, data, top_n):
        words = self.get_words(data)
        tagged_words = nltk.pos_tag(words)
        words = [word.lower() for (word, tag) in tagged_words if tag in ('NN', "NNS", "VB", "VBP", "ADJ", "ADV")]
        fdist = nltk.FreqDist(words)
        return fdist.most_common(top_n)

    def get_nouns(self, data):
        words = self.get_words(data)
        tagged_words = nltk.pos_tag(words)
        words = [word.lower() for (word, tag) in tagged_words if tag in ('NN', "NNS")]
        return words

    def aristo_write_most_common_nouns_to_file(self, data, top_n_most_common, file):
        TextAnalyser._write_tuple_to_file(self.aristo_get_most_common_nouns(data, top_n_most_common), file)

    def aristo_get_similar_sentences(self, main_sentence, list_of_sentences, similarity_score_threshold):
        for sentence in list_of_sentences:
            if TextAnalyser.get_similarity_score(main_sentence, sentence) >= similarity_score_threshold:
                yield sentence

    @staticmethod
    def get_similarity_score(a, b):
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        stopwords.append('')
        tokenizer = WordPunctTokenizer()
        """Check if a and b are matches."""
        tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]

        tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]

        # Calculate Jaccard similarity
        ratio = 0
        if len(set(tokens_a).union(tokens_b)) > 0:
            ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
        return (ratio)


    def Stem_tokens(self, tokens):
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(item) for item in tokens]

    def _normalize(self, text):
        remove_punctuation_map = self._get_punctuation_map()


        tokens = self.Stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
        return  tokens




    def _get_punctuation_map(self):
        result = dict((ord(char), None) for char in string.punctuation)
        result[ord(".")]=" "

        return result

    def does_lists_overlap(self, a, b):
      sb = set(b)
      return any(el in sb for el in a)
