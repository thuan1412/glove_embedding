from pyvi.ViTokenizer import tokenize
import os
import sys
import re

processed_txt = open("./cleaned_text.txt", "w")

DATA_ROOT_NAME = "data/"

topics = os.listdir(DATA_ROOT_NAME)


FLAGS = re.MULTILINE | re.DOTALL


"""
preprocess-twitter.py
`python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

def process_text(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "url")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "user")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes),
                  "smile")
    text = re_sub(r"{}{}p+".format(eyes, nose), "lolface")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes),
                  "sadface")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "neutralface")
    text = re_sub(r"3", "heart")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "number")
    text = re_sub(r"([!?.]){2,}", r"\1 repeat")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 elong")


    return text.lower()


for topic in topics[:1]:
    for file_name in os.listdir(DATA_ROOT_NAME + topic + "/"):
        fulll_name = DATA_ROOT_NAME + topic + "/" + file_name
        file_content = open(fulll_name, "rb").read().decode("utf-16")

        for line in file_content.split('.'):
            line  = process_text(line)
            if line != '\n':
                processed_txt.write(tokenize(line).lower() + "\n")
                
processed_txt.close()