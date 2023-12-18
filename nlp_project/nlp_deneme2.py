import numpy as np
import pandas as pd
from zemberek import (
    TurkishMorphology,
    TurkishTokenizer,
    TurkishSentenceExtractor
)
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

scaler = MinMaxScaler()
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 500)
stop_word_list = nltk.corpus.stopwords.words('turkish')
stop_word_list.extend(["bir", "kadar", "sonra", "kere", "mi", "ye", "te", "ta", "nun", "daki", "nın", "ten"])

morphology = TurkishMorphology.create_with_defaults()
tokenizer = TurkishTokenizer.DEFAULT
extractor = TurkishSentenceExtractor()


def dataCleaning(text):
    """verilen metni token olarak böl ve yalnızca kelimeleri al"""
    tokens = tokenizer.tokenize(text)
    text = [token.content for token in tokens if token.type_.name == 'Word' and token.content.isalpha()]
    sentence = ' '.join(text)
    return sentence


def removeStopwords(text):
    """Zemberek'ten aldığımız stopword kelimelerini kaldırır"""
    #    print(f"Original Text: {text}")
    cleaned_text = [word for word in text if word not in stop_word_list and word not in string.whitespace]
    return cleaned_text


def wordTokenize(text):
    """Önişlenmiş metini kelimelere ayırır ve stopword'leri kaldırır"""
    text = text.split(" ")
    text = removeStopwords(text)
    #    print(text)
    return text


def sentTokenize(text):
    """Önişlenmemiş metini cümlelerine ayırır, bunişlem sırasında önişleme yapar ve stopword'leri kaldırır """
    sent_list = []
    text = text.replace("\"", "")
    results = extractor.from_paragraph(paragraph=text)
    for result in results:
        result_text = dataCleaning(result)
        cleaned_result = removeStopwords(result_text.split(" "))
        if len(cleaned_result) == 0:
            continue
        else:
            sent_list.append(" ".join(cleaned_result))
    #    print(sent_list)
    return sent_list


def lemmas(word_list):
    """Kelime token'larından kök tokenları oluşturur"""
    lemma = []
    for word in word_list:
        analysis = morphology.analyze_sentence(word)
        after = morphology.disambiguate(word, analysis)
        result = [best.get_stem() for best in after.best_analysis()]
        if result == "UNK":
            lemma.append(result)
        else:
            lemma.append(result)
    #    print(lemma)
    return lemma


def wtDist(wt):
    """Kelimelerin dağılımları"""
    flat_list = [item for sublist in wt for item in sublist]
    wt_dist = {}
    for word in flat_list:
        if word not in wt_dist:
            wt_dist[word] = 1
        else:
            wt_dist[word] += 1
    #    print(wt_dist)
    return wt_dist


def wtLenDist(wt):
    """Kelimelerin harf olarak uzunluk dağılımlarını çıkarır ,
    harf uzunluğuna göre kelime sıralamaları yapar ve sözlüğe ekler"""
    wt_len = [len(str(word)) for word in wt]
    wt_len_dist = dict()
    wt_len_dist.fromkeys(range(1, 29))
    for i in range(0, 29):
        wt_len_dist[i] = wt_len.count(i)
    return wt_len_dist


def stLenDist(st):
    """Cümlelerin kelime olarak uzunluk dağılımlarını çıkarır"""
    st_len = [len(wordTokenize(sent)) for sent in st]
    st_len_dist = dict()
    st_len_dist.fromkeys(range(1, 29))
    for i in range(0, 29):
        st_len_dist[i] = st_len.count(i)
    return st_len_dist


data = pd.read_excel("yazar_köşe_yazısı.xlsx")
data.columns = map(str.lower, data.columns)

data['clean_text'] = data['text'].apply(lambda x: dataCleaning(x))
data['word_token'] = data['clean_text'].apply(lambda x: wordTokenize(x))
data['sent_token'] = data['text'].apply(lambda x: sentTokenize(x))
data['lemma_token'] = data['word_token'].apply(lambda x: lemmas(x))
data['wtDist'] = data['lemma_token'].apply(lambda x: wtDist(x))
data['wtLenDist'] = data['word_token'].apply(lambda x: wtLenDist(x))
data['stLenDist'] = data['sent_token'].apply(lambda x: stLenDist(x))

# test ve train verilerinin ayrılması
data['target'], mapping = pd.factorize(data.author)
y = np.array(data.target)
x = data.loc[:, 'text': 'stLenDist']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
a = x_train.reset_index(inplace=True)
b = x_test.reset_index(inplace=True)

# Eğitim ve test veri çerçevelerini birleştirme
combined_df = pd.concat([x_train, x_test], axis=0, ignore_index=True)

# TF-IDF işlemi için kök tokenlarını birleştirme
combined_corpus = [' '.join(map(str, tokens)) for tokens in combined_df.lemma_token]
# Bag of words vektörü
count_vectorizer = CountVectorizer()
combined_sparce_matrix = count_vectorizer.fit_transform(combined_corpus).toarray()
combined_sparce_matrix_normalized = scaler.fit_transform(combined_sparce_matrix)

# tf-idf vektörü
tfidf_vector = TfidfVectorizer()
combined_tfidf_matrix = tfidf_vector.fit_transform(combined_corpus).toarray()

# Kelime uzunluk dağılım vektörü
wtDict_vector = DictVectorizer()
combined_wtLenDist_matrix = wtDict_vector.fit_transform(combined_df.wtLenDist).toarray()
combined_wtLenDist_matrix = scaler.fit_transform(combined_wtLenDist_matrix)

# Cümle uzunluk dağılım vektörü
stDict_vector = DictVectorizer()
combined_stLenDist_matrix = stDict_vector.fit_transform(combined_df.stLenDist).toarray()
combined_stLenDist_matrix = scaler.fit_transform(combined_stLenDist_matrix)

# Attribütleri birleştirme
combined_attribution = (
    combined_tfidf_matrix,
    combined_sparce_matrix,
    combined_wtLenDist_matrix,
    combined_stLenDist_matrix)
combined_attribution = np.concatenate(combined_attribution, axis=1)

# Eğitim ve test veri setlerini tekrar ayırma
X_train_combined = combined_attribution[:len(x_train)]
X_test_combined = combined_attribution[len(x_train):]

# Lojistik Regresyon modelini oluşturun
logistic_model_bow = LogisticRegression(max_iter=10000)

# Eğitim setini kullanarak modeli eğitin
logistic_model_bow.fit(X_train_combined, y_train)

# Test seti üzerinde modelin performansını değerlendirin
y_pred_bow = logistic_model_bow.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred_bow)
conf_matrix = confusion_matrix(y_test, y_pred_bow)
classification_rep = classification_report(y_test, y_pred_bow)

# Performans metriklerini yazdırın
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


