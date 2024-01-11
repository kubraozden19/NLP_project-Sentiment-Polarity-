# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:01:29 2024

@author: Lenovo
"""

import numpy as np
import pandas as pd
import re

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

'''
Burada yapmak istediğimiz kelime sayıları üzerinden olumlu veya olumsuzluğa gitmek, Öncelikle 
biz kelimeleri boşluklarından ayıracak olursak her bir . ) ! gibi ifadeler de kelime sayılacak 
bu yüzden öncelikle cümlelerimizi bu noktalama işaretlerinden temizlemeliyiz. 
Burada spars matrix kavramından da söz edelim. Biz kelime vektörü oluşturacağız yani mesela birinci
yorumda "Wow... Loved this place." 4 tane kelime var bu 4 kelimenin hiçbirisi  2. yorumda "crust is not good."
tekrar etmiyor veya 3. cümlede tekrar etmiyor sonraki cümlelerde mesela "the" kelimesi tekrar ediyor
ondan sonraki "the" kelimesi daha önceden geçen "the" kelimesi ile eşleşiyor oradan bir yakalayacak bu arada "the"
kelimesi stop word yani anlam ifade etmeyen kelime dolayısıyla burada yine hiçbir eşleşme olmayacak.
Dolayısıyla uzayımız yani her bir yorum için yazılan kelime vektörü giderek büyüyecek ve kelime vektörünün
büyük bir kısmının boş olacağını öngerebiliyoruz. Kelimelerin toplamda kaç yorumda geçtiğini saydığımızda 4 ü 5 i
geçmeyecek. Buna genel olarak spars matrix deniliyor yani matrisin çok büyük bir kısmının boşluktan oluşması durımu
Mesela bütün kelimelerden hangileri bu yorumda geçiyor diye bakarsak geçen kelimeleri 1 ile işaretlediğimizde ki 
bu bin yorumda toplamda 10 bin kelime geçiyor olsa bu 10 bin kelimeden sadece 10 tanesi falan burada geçiyor olacak
dolayısıyla 9990 tane 0, 10 tane 1 olan bir kelime vektörüne ulaşıyor olacağız. 

'''
# İmla işaretleri ve Alfanümerik verilerin Filtrelenmesi------------------------------------------

# yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][0]) 
# küçük harfler ve büyük harfler içermeyenleri filtrele ve bunları boş karakter ile değiştir.

# Büyük küçük harf problemi-------------------------------------------------------------------
# Her zamn böyle olur gibi düşünmeyin bazen büyük küçük harf yazılmasının arasında problem açısından farklılık olabilir
# öznitelik çıkartmak açısından farklılık olabilir. Biz sentiment polarity çalıştığımız için bunların birbirine
# çevrilmesi gerekiyor çünkü bu kelimenin büyük harfle veya küçük harfle başlaması aslında anlam olarak çümleye yaptığı
# etki olarak duygusal olarak pozitiflik- negatiflik açısından bir fark ifade etmiyor dolayısıyla bunların hepsini standart 
# hale çevirmeliyiz.

# Feature Extraction (Öznitelik Çıkarımı)
# Bag of words (BOW)
 
# Kelime Gövdelerinin Bulunması (Stemmer)
import nltk

from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()

# Stop Word (Durma Kelimeleri)
nltk.download('stopwords') # stopwords kelimeler makine öğrenmesine dahil edilemeyecek o yüzden onları metinden ayıklayacağız
from nltk.corpus import stopwords 

derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i]) 
    yorum = yorum.lower() # büyük harfleri küçük harflere çevirelim
    yorum = yorum.split() # yorumu bir liste haline getirelim
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] 
    
    '''
    Kelimeleri tek tek alıp stopword olup olmadığına bakacak eğer stopwords değilse o zaman kelimenin kökünü bulacak
    ve yeni listenin ilke elemanı olarak atayacak, sonra diğer kelimeye bakacak eğer stopwords ise o kelimeyi atlayacak, 
    diğer kelimelere geçecek vb şekilde ilerleyecek
    
    '''
    yorum = ' '.join(yorum) # liste olan yorumu stringe çevirdik
    derlem.append(yorum)

# Kelime Vektörü Sayaç kullanımı (CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000) 
# max_features : en çok kullanılan 2000 kelimeyi al daha fazlasını alma dedik sınırladık

X = cv.fit_transform(derlem).toarray()

yorumlar['Liked'].fillna(0, inplace = True) # liked sütunundaki olmayan verilere 0 atadık
Y = yorumlar.iloc[:, 1].values

# X : bağımsız değişken
# Y : bağımlı değişken

'''
X'e baktığımızda buradakilerin hepsi birer kelime yanı 0. 1. kelime gibi bu kelimelerin her birisi için 
bu kelimenin hangi yorumlarda geçtiğini görebiliyoruz. Bir vektörümüz oluştu bu kelime vektörü burada geçen 
2000 kelimenin tamamı x ekseninde, 1000 tane yorumun tamamı da y ekseninde. Her kelime için o yorumda var mı yok mu ?
yani spars matrix dediğimiz böyle bir şey çoğunluğu boş matris demek, bu matrisin çoğunluğunun boş olma nedeni
çünkü çoğu kelime çoğu yorumda geçmiyor mesela bi yorumda geçen kelime başka hiçbir yorumda geçmemiş olabilir. 
yorumlardaki cümleler yani her bir kelime artık sayısal değerlere dönüştü ve 2000 tane özellik, öznitelik olarak dönüştü.
Yani tek bir kolon olan 'Review' kolonu bir anda 2000 kolona çıktı her kelime için var yok anlamında kolonlar oluştu. 
Her satır için 2000 tane öznitelik çıkartmış olduk. 
'''




# Buraya kadar öznitelik çıkarma işlemleri yaptık artık MAKİNE ÖĞRENMESİ kısmına geçelim


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 




