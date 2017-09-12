from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

doc1 = "Phone is good, it gets warm while gaming. Overall performance is good."
doc2 = "Camera is not that good. Daylight images are quite ok. No heating issues. Good display and battery."
doc3 = "No heating issues and fast charger works fine. Camera quality could have been better, but performrd fairly."
doc4 = "Good phone with this price tag. Mi have to work on camera quality and display pixcel quity.... Remaining is good.... Battery back up is awsm... Overall good phone to daily use.... Not for too much heavy use cz processor is average."
doc5 = "though is camera is avg as compare with others. and lastly the battery is excellent ."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]




stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# Reading each doc in doc_complete
doc_clean = [clean(doc).split() for doc in doc_complete]  

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# To print the cleaned data
print(doc_clean)

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
# print(ldamodel.print_topics(num_topics=5, num_words=5))
