#1.Eşik değer belirle.
#2.Aykıır değerlere eriş.
#3.Hızlıca aykırı değer var mı yok sorgula.


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#pip install -U seaborn
#! pip install --upgrade matplotlib
#!pip install missingno

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.float_format", lambda x: "%.3f" %x)

#büyük ölçekli uygulamalarımız için application_csv dosyasını okutuyoruz
def load_application_train():
  data=pd.read_csv("C:/Users/esman/PycharmProjects/feature_engineering/datasets/application_train.csv")
  return data

df = load_application_train()
df.head()

#küçük ölçekli işlemlerimiz için titanic veri setini okutuyoruz. önce küçük veride bakıcaz sonra büyük veri setinde nasıl durduğunu gözlemleyeceğiz.
def load():
  data = pd.read_csv("C:/Users/esman/PycharmProjects/feature_engineering/datasets/titanic.csv")
  return data

df = load()
df.head()


### 1. OUTLİERS (Aykırı Değerler)
#AYKIRI DEĞERLERİ YAKALAMA
#boxplot (kutu grafik); bir sayısal değişkenin dağılım bilgisini verir. en yaygın olarak boxplot ya da histogram grafiği kullanılır.
#Grafik Teknikle Aykırı Değerleri gözlemlemek istediğimizde boxplant yöntemi kullanılır.

__version__ = "0.12.2"

sns.boxplot(x=df["Age"])
plt.show()

#eşik değerlere erişmek için çeyrek değerler bulunur.
#Aykırı Değerler Nasıl Yakalanır?
q1 = df["Age"].quantile(0.25)  #q1 = 20.125
q3 = df["Age"].quantile(0.75)  #q3 = 38.0

#çeyrek değerler üzerinden iqr hesabı yapılır
iqr=q3-q1
up  = q3 + 1.5*iqr   #yaş değişkeni için üst sınırım = up  = 64.8125
low = q1 - 1.5*iqr   #yaş değişkeni için alt sınırım = low = -6.6875 --> yaş değişkeninde eksik değerler olmadığı için alt sınırım eksi geldi.

#belirlediğimiz eşik değerlere göre aykırı değerler barındıran satırlar
df[(df["Age"] < low) | (df["Age"] > up)]

#aykırı değerleri barındıran satırların sadece index bilgisini getir.
df[(df["Age"] < low) | (df["Age"] > up)].index

#Aykırı Değer Var mı Yok mu? axis=none yaptım ki satır sütun hepsini incelesin diye.
#bu değişkenlerde aykırı değerler olduğunu görüyorum çünkü true çıktısı veriyor bana.
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

#true çıktıısnı doğrulatmak için ~ ile dışındaki seç ve var mı diye sorgulat. yine true döner.
df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

#low değişkeninde eksik değer yoktu. burada aykırı değer yok bu nedenle false döner.
df[~(df["Age"] < low)].any(axis=None)


#### FONKSİYONLAŞTIRMA
#bu işlemleri fonskiyonlaştırmak istiyorum.

def outlier_thresholds(dataframe,col_name, q1=0.25, q3=0.75):
  quartile1 = dataframe[col_name].quantile( q1 )
  quartile3 = dataframe[col_name].quantile( q3 )
  interquantile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquantile_range
  low_limit = quartile1 - 1.5 * interquantile_range
  return low_limit, up_limit

#yaş değişkeni için alt ve üst değerleri belirlerdi.
outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

#low,up değerlerini tut.
low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()

df[(df["Fare"] < low) | (df["Fare"] > up)].index


#aykırı değer var mı yok mu bilgisini fonksiyonlaştırarak yaz.
# check_outlier fonksiyonu --> aykırı değerler var mı? Bool tipinde çıktı vermesi önemli
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True  #aykırı değer varsa true döner.
    else:
        return False #aykırı değer yoksa false döner.

#100 tane değişken olsaydı böyle tek tek mi fonksiyonu çağıracaktım. tek tek değişken özelinde mi çağıracağım
# hayır. onun için de birazdan bir fonksiyon tanımlayacağım.
check_outlier(df, "Age")
check_outlier(df, "Fare")


#öyle bir fonksiyon tanımlamalıyım ki bu fonksiyon veri setindeki sayısal değişkenleri otomatik olarak seçiyor olsun. aynı zamanda diğer değişkenleri de ayrı ayrı seçecek.
# grab_col_names fonksiyonu: docsting yazdık bak oraya.
###################

dff = load_application_train()
dff.head()

#fonksiyona argümanlar girmişim cat_threshold=10 yani eğer bir değişkenin 10 dan az sınıfı varsa bu benim için kategorik bir değişkendir. diyorum.
#car_threshold=20. eğer bir kategorik değişkenin 20 den fazla sınıfı varsa

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]               #önce kategorik değişkenler seçilir

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and    #numerik ama kategorikler seçilir.
                   dataframe[col].dtypes != "O"]

    # eğer bir kategorik değişkeni 20 den büyükse aynı zamanda tipi de kategorikse bu değişken kategorik değildir. kategorik gibi duruyorsun ama kardinel. yani ölçülebilirliği yok. eşsiz çok fazla sınıfın var.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

#şimdi bu fonksiyonları veri setine uygula sana göndersin sonuçları:
for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")

#index bilgini de istiyorsak ön tanımlı değeri true olarak gönder.
grab_outliers(df, "Age", True)

age_index = grab_outliers(df, "Age", True)

#şimdiye kadar hangi fonksiyonları tanımladık;
outlier_thresholds(df, "Age")
check_outlier(df, "Age")         #sadece bir değişkende aykırı değer var mı?
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme
############################################
#bir yöntem olarak aykırı değerleri silebilirsin.

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]
#bu şekilde silme yöntemiyle aykırı değerlerden kurtulduğumuzda diğer tam olan gözlemlerden de oluyoruz. bu nedenle bu değerleri silmek yerine baskılamayı da uygulayabiliriz.

# 2. Baskılama Yöntemi (re-assignment with thresholds)
#aykırı değerler yakalanır. ardından bu değerler eşik değerler ile değiştirilir.
low, up = outlier_thresholds(df, "Fare")


#sseçim işlemini iki şekilde yapabilirsin. ya böyle df üzerinden seçip değişkeni yolla.
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
#ya da loc üzerinden seçim işlemlerini gerçekleştir.
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

#üst sınıra göre aykırı olan değerleri getirir.
df.loc[(df["Fare"] > up), "Fare"] = up

#alt sınıra göre aykırı değerlerim yoktu ancak bunu da sorgulayalım yine de.
df.loc[(df["Fare"] < low), "Fare"] = low

#şimdi bunu programlaştırıp fonksiyon üzerinden gerçekleştirelim işlemleri.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

#outlier var mı yok mu?
for col in num_cols:
    print(col, check_outlier(df, col))

#her bir thresholdla değişkenin değerleirni değiştirelim.
for col in num_cols:
    replace_with_thresholds(df, col)

#tekrar soralım aykırı değer var mı?
for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Recap
###################

df = load()   #veri setini okuduk
outlier_thresholds(df, "Age")  #aykırı değerleri saptama
check_outlier(df, "Age")       #tresholdlara göre outliers var mı yok mu?
grab_outliers(df, "Age", index=True) #outliers index bilgisini getir.

remove_outlier(df, "Age").shape  #silme yöntemi
replace_with_thresholds(df, "Age") #baskılama yöntemi
check_outlier(df, "Age")



#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################



#tek başına aykırı olmayan değerler bir arada inceledndiğinde aykırı olabilir.
# yani mesela 17 yaşında olmak normal.
#3 kere evlenmek de normal.
#ama 17 yaşında olup 3 kez evlenmiş olmak normal değil, aykırı değerdir. buna çok değişkenli aykırı değer analizi denir.
# 17, 3

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

#aykırı değerler var mı?
for col in df.columns:
    print(col, check_outlier(df, col))

#aykırı değerlerden bir değişkeni seçtim.
low, up = outlier_thresholds(df, "carat")
#carat değişkeninde kaç tane outlier vardır.
df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape


#komşuluk sayımı 20 olarak giriyorum.
clf = LocalOutlierFactor(n_neighbors=20)
#lof skorlarını getirecek
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

# df_scores = -df_scores
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]

df[df_scores < th].shape


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)



#############################################
# Missing Values (Eksik Değerler)
#############################################

# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi yani dolu
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

#eksik değer sayısını veri setindeki tüm gözlemlere bölüp 100 ile çarptığında. tüm veri setinde ki eksik değre oranını bulmuş olacaksın.
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# listcomperhension yapısı ise df.columns da gez ve eksik değer sayısı 0 dan büyük ise bu col adını seç diyorum. böylece eksik değer olan değişkenlerin sadece adları gelecek.
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

#şimdi de bu eksik değerler için yaptığımız işlemleri fonksiyonlaştıralım.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    #eksik değer sayısı için;
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    #eksik değer oranı için;
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    #axis=1 ,sütunlara göre birleştirme işlemi yapmak istiyorum.
    #np.round(..) virgülden sonraki basamakla ilgili bir ayarlama yapılmış.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)

# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum() #yaş değişkenini onun ortalaması ile doldurdum.
df["Age"].fillna(df["Age"].median()).isnull().sum() #medyanı ile doldurdum.
df["Age"].fillna(0).isnull().sum() #istediğin bir değer ile boşlukları doldurabilirsin.

#axis=0 satırlara göre gider yani aslında konuya sütun bazında bakar.
#bunu yaptığında hata alırsın çünkü veri setinde hem kategorik hem de numerik değişkenler var.
#bu hatayı almamak için bunlardan sadece sayısal değişkenleri seçip doldurmalısın
# df.apply(lambda x: x.fillna(x.mean()), axis=0)


#x.dtype != "O" eğer ilgili değişkenin tipi objecten farklı ise; bu değişkeni ortalaması ile doldur, değilse de olduğu gibi kalsın istiyorum.
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

#değişkendeki eksik verileri ortalaması ile doldurur.
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

#kategorik değişkenler için en mantıklı yöntem modunu almak.
#embarked değişkenindeki eksiklikleri modunu alarak doldur.
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
#embarked değişkenindeki eksiklikleri bir string değer ile doldur. Burada "missing" ile doldur demişim.
df["Embarked"].fillna("missing")
#unique , eşsiz değer sayısı
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama

#cinsiyete göre groupby işlemini gerçekleştir. bunun içinden yaş değişkenini seç ve ortalamasını al.
df.groupby("Sex")["Age"].mean() #kadınların yaşlarının ortalaması 27.916, erkeklerin yaşlarının ortalaması 30.727

#yaşın ortalamsını aldık 29
df["Age"].mean()

# şimdi eksik değerleri dolduruken direkt yaşın ortalaması olan 29la doldurmaktansa cinsiyet özelinde ortalama ile doldurursan daha mantıklı olacaktır.
#aşağıdaki kod satırının daha açık hali bundan sonraki 3.kod satırına bak.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

#cinsiyet bazında ortalama alır ve eksik değerleri doldurur.
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()



#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#passengerId çıkar.
num_cols = [col for col in num_cols if col not in "PassengerId"]
#drop_first=True, iki sınıfa sahip olan kategorik değişkenlerde ilk sınıfı silecek. ikinci sınıf kalacak.
#böylece 2 sınıf veya daha fazla sınıfa sahip olan kategorik değişkenleri numerik bir şekilde ifade etmiş oluruz.
#get_dummies fonksiyonu sadece kategorik değişkenlere bir dönüşüm uygulamaktadır.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması, makine öğrenme modelini uygulayabailmek için veriyi uygun hale getirmelisin.
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması. KNN tahmine dayalı bir şekilde eksik değerleri doldurmamızı sağlayacak.
from sklearn.impute import KNNImputer
#Gözlemin en yakın 5 komşusuna bakar ve onların ortalamsıyla boşluğu doldurur.
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
#standartlaşma işleminden önceki haline döndüüryorum.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma


#############################################
# Gelişmiş Analizler


# Eksik Veri Yapısının İncelenmesi, bağımlılık durumu var mı yok mu?
#eksik değerler ile bağımlı değişken arasındaki ilişkiyi inceleyen fonksşyon yazacağız.
#veri setinde eğer birden fazla değişkende eksiklik varsa o zaman bu eksik değrler acaba birbirlerinden mi etkilenerek ortaya çıktı? ysni belirli bir değişkene bağlı olarak ortaya çıktı.
# 2 yol var. ya eksiklikler birlikte ortaya çıktı ya da eksiklikler başka bir değişkenden etkilerek ortaya çıktıç
# R Methodu = ilgili veri setindeki değişkenlerdeki tam sayıları göstermektedirç
msno.bar(df)
plt.show()
#MATRİX Methodu = değişkenlerdeki eksikliği incelememizi sağlayan görsel bir araçtır. değişkenlerdeki eksikliğin birlikte çıkıp çıkmadığı hakkında bilgi verir.
msno.matrix(df)
plt.show()
#HEATMAP Methodu = eksiklikle üzerine kurulu bir ısı haritasıdır. çok değerlidir.
msno.heatmap(df)
plt.show()
#şimdi bu ısı haritasına baktığımızda 0.1 ve -0.1 değerleri var. bunlar anlamlı değil. bu eksiklikler demek ki birlikte ortaya çıkmamış.
# eğer 1 e yakın olsaydı pozitif korelasyon. -1 e yakın olsaydı negatif korelasyon olduğunu söyleyecektim.






###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi


missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

#survived değişkeni ile na_cols(eksikliği olan değişkenleri) karşılaştırmasını istiyorum, missing_Vs_target fonksiyonu ile.

missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)





#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding (2 den fazla sınıf varsa) & Binary Encoding (2 sınıf varsa)

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
#fit_transform ile SEX değişkenine LabelEncoder uyguluyorum
#0 dan 5 e kadar labells numara verdim
le.fit_transform(df["Sex"])[0:5]
#bu verdiğim numaraları unutursam inverse_tranform ile öğrenebilirim.
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()
#eğer burada unique methodunu kullanıp len alırsanız eksik değerleride sınıftan sayar. bu nedenle nunique kullanılır.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

#yukarıdaki işlemin aynısını application_train veri setine uygula.
df = load_application_train()
df.shape
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

#karıştırılan yerler, dikkat et.
df = load()
df["Embarked"].value_counts()    #embarked kaç sınıfı var? =3
df["Embarked"].nunique()         #embarked eşsiz değerlerini getirir?
len(df["Embarked"].unique())     #eşsiz değerleri getiri. len ile yapınca eksik değeride sayar ve 4 çıktısını verdi.

#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
#embarked değişkeninin 3 tane sınıfı(S,C,Q) var. bu sınıflar arasında sıralama fark yok yani takımlarda ki gibi .
df["Embarked"].value_counts()

#get_dummies methoduna bir df verilir, ve dönüştürmek istediğin sütunların adı girilir. sadece onlar dönüştürülür diğerlerine dokunulmaz.
pd.get_dummies(df, columns=["Embarked"]).head()

#drop_first true yapıldığında ilk sınıf (   C   ) uçtu.
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

#Eğer ilgili değişkendeki eksik deeğerlerde bir sınıf/değişken olarak gelsin istiyorsam dummy_na=True girilir.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

#2 sınıf kategorik değişken için hem binary encoding işlemini hem de one hot encoding işlemini yapıyor oluyorum.
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#ohe işlemine uygulanacak değişkenleri seçmek için şart koyuyorum.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()

df.head()

#############################################
# Rare Encoding

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getirmesi için bir fonksiyon tanımlıyorum.
#eğer frekansları görselleştirmek istiyorsak plot=true girilir.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

#elinde bol kategorik değişkenli bir veri seti varsa rare_analyser fonks. kullan.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()





#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

# StandardScaler: Klasik standartlaştırma- Normalleştirme.
# Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s  (Z Standartlaştırılması)


df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


#Başka bir standartlaştırma yöntemi --> ROBUST SCALER
# RobustScaler: Medyanı çıkar iqr'a böl. StandartScaler a göre aykırı değerlere karşı daha dayanıklıdır. yani aykırı değerlerden etkilenmez.
#Standart Scaler kullanmak yerine RobustScaler kullanmak daha mantıklı olacaktır.

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#Minmax Scaler de standartlaşma yöntemidir.
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
#qcut fonskiyonu bir değişkenin değerleirni küçükten büyüğe sıralar ve çeyrek değerlere göre x parçaya (biz 5 girdik) böler

df["Age_qcut"] = pd.qcut(df['Age'], 5) #kendi istediğin labellar varsa labels argümanıda tanımlayabilirsin.




#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################
#binary encoding ile karıştırma farklı şeyler. burada var olanları değiştiriyordun.
# Binary Features: Flag, Bool, True-False (burada ise var olanlardan yeni değişkenler türetiyorsun)

df = load()
df.head()

#dolu olanlara1, boş olanlara 0 yazar.
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

#Survived ortalamsaını alalım::
# new_cabin_bool=0 olanların survived değeri= 0.300
# new_cabin_bool=1 olanların survived değeri= 0.667
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

#2 grubu kıyaslama için oran testi yapalım. oran testi için 2 parametreye ihtiyacım var. count (başarı sayısı yani hayatta kalma) ve nobs (gözlem sayısı)
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),   #kabin numarası olanlardan kaç kişi hayatta kalmış.
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],  #kabin numarası olmayanlardan kaç kişi hayatta kalmış

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],  #kabin numarası olanlar kaç kişi
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]]) #kabin numarası olmayanlar kaç kişi

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # p_value=0.00 çıktı 0.05 den kçk olduğu için H0 reddedilir. yani anlmalı fark var.

#başka bir binary feature oluşturalım.
#kişilerin gemideki akrablıklarını ifade eden 2 değişken var. SibSp (yakın akraba) ve Parch (uzak akraba)
#BELKİ GEMİDEKİ kişilerin yakınlarının yanında olması durumu onun hayatta kalma ihtimalini değiştirmiştir diye düşünüyorum. ve bunu inceliyorum.
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"   #bu iki değişkenin toplamı 0 dan büyükse new_is_alone adında yeni bir değişken oluştur. ve NO değerini ata. (hayır yalnız değil yani yanında ailesi var)
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES" #bu iki değişkenin toplamı 0 a eşitse new_is_alone adında yeni bir değişken oluştur. ve YES değerini ata. (evet, yalnız yani yanında ailesi yok.)

#survived değişkeninde incele bakalım.
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"}) #(no=0.506 iken yes=0.304 yani yalnız olanların hayatta kalma ihtimali daha düşük çıkmış)

#Birde buna HİPOTEZ TESTİ uygulayalım.
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  #p_value=0.00 H0 hipotezi reddedilir. yani fark varmış, bu etkiyi göz ardı edemeyiz.

#############################################
# Text'ler Üzerinden Özellik Türetmek

df.head()

###################
# Letter Count = harfleri saydırma
###################
#df içindeki name değişkenindeki  tüm stringleri say. bir değişkende kaç tane harf varsa bunu saymış olur.
df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count = kelimeleri sayalım.
###################
#str(x)= ismi komple bir stringe çevirsin.  Ve boşluklara göre(" ") split etmesini istiyorum. böylece len ile kaç kelime olduğunu buluyorum.
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################
# name değişkeninde ilgili satırı split edeceğiz. ardından for ile bunlarda gezeceğiz. eğer gezdiğin her kelimenin başında Dr varsa bunu seç (x) daha sonra bunun len al.
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
#dr kategorik bir değişken olduğu için liste içerisinde ortalamayı sorgulatmanın yanı sıra countu da gir.
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################
#REGEX = RegularExpretion
df.head()
#df.Name yerine df["Name"] de yazabilirdin.
#.extract = çıkar demektir. neyi çıkarıcak --> (' ([A-Za-z]+)\.', expand=False) --> yani önünde boşluk,sonunda nokta olacak arada ise büyük veya küçük harflerden oluşacak ifadeleri yakalamasını istiyorum.
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
#info aldığımızda Timestamp değişkeninin türünün object olduğunu görüyorum. to_datetime ile dönüşüm işlemlerini gerçekleştiriyorum
dff.info()
#format ile timestamp içinde yer alan değişkenlerin sırası verilir. yıl-ay-gün
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı*12 + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date içinde birsürü method var bak onlara.


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load()
df.head()
#yaptığın bu işlemler bir şeyi ifade etmeli. mesela age ile pclass çarpmışsın ama neden? buradan çıkan sonuçla şu çıkarımı yapabilirsin;
#yaşı küçük olmsına rağmen 1.sınıf yolculuk yapıyor, yaşı büyük olmasına rağmen 3.sınıf yolculuk yapıyor gibi refah düzeylerinden sonuçlar çıkarabilirsin.
# burade age değişkenini bir standartlaşmaya sokmam gerekir.
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

#art 1 kişinin kendisini akrabalrının sayısına ekleyince ailedeki kişi sayısnı ver.
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()
#kadınların hayatta kalma oranları zaten daha yüksekdi ama şimdi bunuda standartlaştırdık. ve genç kadınların yaşlılara göre daha yüksek oranda hayatta kaldıklarını gördük.
df.groupby("NEW_SEX_CAT")["Survived"].mean()


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()
#değişkenlerin isimlerini büyütüyorum
df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
############################################

#Türettiğimiz yeni değişkenlerin hepsini getir;

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

# değişken tiplerini çağır.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]


#############################################
# 2. Outliers (Aykırı Değerler)
#############################################
#outliers kontrol etmek için "check_outlier" fonksiyonunu çağırıyorum.
for col in num_cols:
    print(col, check_outlier(df, col))
#eşik değerlerle bu aykırı değerleri değiştirmek istiyorsam; replace_with_thresholds fonksiyonu çağır.
for col in num_cols:
    replace_with_thresholds(df, col)
#tekrar aykırı değerlere bakalım kurtuldun mu gözlemle. kurtuldun.
for col in num_cols:
    print(col, check_outlier(df, col))


#############################################
# 3. Missing Values (Eksik Değerler)
#############################################
#eksik değer tablomuz;
missing_values_table(df)
#cabin_bool adında yeni bir değişken oluşturmuştuk bu nedenle CABIN değişkenini kalıcı (inplace=True) olarak sil.
df.drop("CABIN", inplace=True, axis=1)

#ticket ve name değişkenlerinide uçur.
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

#yaş değişkenindeki eksiklikleri medyani ile doldur. Age değişkenindeki eksikliklerden kurtuldun.
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

#AGE ile oluşturduğun yeni değişkenlerdeki eksiklikler ne olacak?
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

#tekrar eksik değer tablosuna bak. yeni age ile oluşturduğun değişkenlerdeki eksikliklerde gitti.
missing_values_table(df)

#embarked değşikenindeki eksikliklerden de kurtuluyorum.
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#tekrar eksik değer tablona bak, uçmuş mu?
missing_values_table(df)

#############################################
# 4. Label Encoding
#############################################
#2sınıflı kategorik değişkenleri seç
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################
#onehorencoding işlemimne tabi tutulacak sütunları seç
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
#seçtiğin sütunları onehotencoder dan geçir.
df = one_hot_encoder(df, ohe_cols)
#böylece bütün olası kategoriler değişkenlere dönüştü.
df.head()
df.shape

#evet ben yeni değişkenler oluşturdum ancak bunlar ne kadar anlamlı. bunun için ilk olarak grab_col_names çağırıyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

#anlamlı olmayan kullanışsız sütunlara bir bak.
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
#kullanışsız olan bu sütunları silebilirsin.
# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################
# y = bağımlı değiken // x= bağımsız değişkenleri (survived ve passengerıd dışındakiler) tanımlıyorum.
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
#train seti üzerinde model kuracağım
#test seti ile kurduğum modeli test edeceğim.
#yani train_test_split yaklaşımını kullanacağım.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

#modeli kur
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
#bu modeli kullanarak tahmin edelim o zaman diyerek x_test (bağımsız değişken değerlerini) modele sor.
y_pred = rf_model.predict(X_test)
#test setinin y bağımlı değişkeni ile tahmin ettiğin değerleri(y_pred) kıyasla.
accuracy_score(y_pred, y_test)

#############################################################
# bizim en zor işimiz veriyi ön işlemeden geçirmekti, model kurmak kolay. yeni değişkenler vs üretme kısmı zaman alan yer.
# Hiç bir işlem yapılmadan model kursaydık elde edilecek skor ne olurdu?

dff = load()
#€ksik değerleri drop etmeseydin randomforest çalışmazdı.
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde? bunlar anlamlı mı anlamsız mı?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
#çıkan grafikte hangi değişkenin en anlamlı olduğunu görüyorsun.
