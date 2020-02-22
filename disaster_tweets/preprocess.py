import sys
import pandas as pd
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


def token_lemmatize(sentence):
    return " ".join([wnl.lemmatize(text) for text in sentence.split(' ')])


def read_file(fname, out_fp):
    df = pd.read_csv(fname)
    regexs = [
        "\r\n|\n",
        "https?://[0-9a-zA-Z./]*",
        "[^A-Za-z']",
    ]
    values = [
        " ",
        " ",
        " ",
    ]
    df["location"].replace(regex="\r\n|\n", value="", inplace=True)
    df["text"].replace(regex=regexs, value=values, inplace=True)
    df["text"].replace(regex="'{2,}", value="'", inplace=True)
    df["text"].replace(regex="\s'|^'|'\s|'$", value=" ", inplace=True)
    df["text"].replace(regex="\s{2,}", value=" ", inplace=True)
    df["text"] = df["text"].str.lower().str.strip()
    df["text"] = df["text"].apply(lambda x: token_lemmatize(x))
    replace_regex = [
        "wa ", "ha ", "doe ",
        "it's", "he's", "she's", "this's", "that's", "there's", "here's", "who's", "what's", "let's",
        "'m", "'re", "'ve", "'d", "'ll",
        "aren't", "don't", "didn't", "doesn't", "isn't", "wasn't", "weren't", "can't", "haven't", "hasn't", "shouldn't"
        "'s been",
    ]
    replace_value = [
        "was ", "has ", "does ",
        "it is", "he is", "she is", "this is", "that is", "there is", "here is", "who is", "what is", "let us",
        " am", " are", " have", " would", " will",
        "are not", "do not", "did not", "does not", "is not", "was not", "were not", "can not", "have not", "has not", "should not"
        " has been",
    ]
    df["text"].replace(regex=replace_regex, value=replace_value, inplace=True)
    df["text"].replace(regex="'s", value=" 's", inplace=True)

    df.to_csv(out_fp, index=False, sep='\t')


def main():
    train_out_fp = open("data/train.csv.formatted", 'w', encoding="utf-8", newline='\n')
    read_file("data/train.csv", train_out_fp)
    train_out_fp.close()

    test_out_fp = open("data/test.csv.formatted", 'w', encoding="utf-8", newline='\n')
    read_file("data/test.csv", test_out_fp)
    test_out_fp.close()

def debug():
    read_file("data/debug", sys.stdout)

if __name__ == "__main__":
    main()
    # debug()