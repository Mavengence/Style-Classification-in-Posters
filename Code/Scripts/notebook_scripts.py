from nltk.corpus import stopwords


german_stop_words = stopwords.words('german')
german_stop_words.append("uhr")
german_stop_words.append("the")

def split_array(x):
    try:
        return len(x.split(" "))
    except:
        return 1
    
def stop_word_removal(x):
    token = x.split()
    return ' '.join([w for w in token if not w in german_stop_words])