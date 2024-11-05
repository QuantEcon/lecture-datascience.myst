---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Working with Text

**Author**
> - [Paul Schrimpf *UBC*](https://economics.ubc.ca/faculty-and-staff/paul-schrimpf/)

**Prerequisites**

- {doc}`Visualization Rules <../tools/visualization_rules>`
- {doc}`Regression <../tools/regression>`
- {doc}`Classification <../tools/classification>`
- {doc}`Maps <../tools/maps>`

**Outcomes**

- Use text as features for classification
- Understand latent topic analysis
- Use folium to create an interactive map
- Request and combine json data from a web server


```{literalinclude} ../_static/colab_full.raw
```

## Introduction

Many data sources contain both numerical data and text.

We can use text to create features for any of the prediction methods
that we have discussed.

Doing so requires encoding text into some numerical representation.

A good encoding preserves the meaning of the original text, while
keeping dimensionality manageable.

In this lecture, we will learn how to work with text through an
application --- predicting fatalities from avalanche
forecasts.

## Avalanches

Snow avalanches are a hazard in the mountains. Avalanches can be
partially predicted based on snow conditions, weather, and
terrain. [Avalanche Canada](https://www.avalanche.ca/map) produces
daily avalanche forecasts for various Canadian mountainous regions.
These forecasts consist of 1-5 ratings for each of three
elevation bands, as well as textual descriptions of recent avalanche
observations, snowpack, and weather. Avalanche Canada also
maintains a list of [fatal avalanche incidents](https://www.avalanche.ca/incidents/) . In this lecture, we will
attempt to predict fatal incidents from the text of avalanche
forecasts. Since fatal incidents are rare, this prediction task will
be quite difficult.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


%matplotlib inline
```

### Data

Avalanche Canada has an unstable json api. The api seems to be largely
tailored to displaying the information on various Avalanche Canada
websites, which does not make it easy to obtain large amounts of
data. Nonetheless, getting information from the API is easier than
scraping the website. Generally, whenever you're considering scraping
a website, you should first check whether the site has an API available.

#### Incident Data

```{code-cell} python
# Get data on avalanche forecasts and incidents from Avalanche Canada
# Avalanche Canada has an unstable public api
# https://github.com/avalanche-canada/ac-web
# Since API might change, this code might break
import json
import os
import urllib.request
import pandas as pd
import time
import requests
import io
import zipfile
import warnings


# Incidents
url = "http://incidents.avalanche.ca/public/incidents/?format=json"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode('utf-8'))
incident_list = result["results"]
while (result["next"] != None):
    req = urllib.request.Request(result["next"])
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    incident_list = incident_list + result["results"]
incidents_brief = pd.DataFrame.from_dict(incident_list,orient="columns")
pd.options.display.max_rows = 20
pd.options.display.max_columns = 8
incidents_brief
```

```{code-cell} python
# We can get more information about these incidents e.g. "https://www.avalanche.ca/incidents/37d909e4-c6de-43f1-8416-57a34cd48255"
# this information is also available through the API
def get_incident_details(id):
    url = "http://incidents.avalanche.ca/public/incidents/{}?format=json".format(id)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    return(result)


incidentsfile = "incidents.csv"

# To avoid loading the avalanche Canada servers, we save the incident details locally.

try:
    incidents = pd.read_csv(incidentsfile)
except Exception as e:
    incident_detail_list = incidents_brief.id.apply(get_incident_details).to_list()
    incidents = pd.DataFrame.from_dict(incident_detail_list, orient="columns")
    incidents.to_csv(incidentsfile)
incidents.head()
```

Many incidents include coordinates, but others do not. Most
however, do include a place name. We can use [Natural Resources Canada's
Geolocation Service](https://www.nrcan.gc.ca/earth-sciences/geography/topographic-information/geolocalisation-service/17304)
to retrieve coordinates from place names.

```{code-cell} python
# geocode locations without coordinates
def geolocate(location, province):
    url = "http://geogratis.gc.ca/services/geolocation/en/locate?q={},%20{}"
    req = urllib.request.Request(url.format(urllib.parse.quote(location),province))
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    if (len(result)==0):
        return([None,None])
    else:
        return(result[0]['geometry']['coordinates'])
if not "alt_coord" in incidents.columns:
    incidents["alt_coord"] = [
        geolocate(incidents.location[i], incidents.location_province[i])
        for i in incidents.index
    ]
    incidents.to_csv(incidentsfile)
```

Now that we have incident data, let's create some figures.

```{code-cell} python
# clean up activity names
incidents.group_activity.unique()
```

```{code-cell} python
incidents.group_activity=incidents.group_activity.replace("Ski touring","Backcountry Skiing")
incidents.group_activity=incidents.group_activity.replace("Out-of-Bounds Skiing","Backcountry Skiing")
incidents.group_activity=incidents.group_activity.replace("Lift Skiing Closed","Backcountry Skiing")
incidents.group_activity=incidents.group_activity.replace("Skiing","Backcountry Skiing")
incidents.group_activity=incidents.group_activity.replace("Snowshoeing","Snowshoeing & Hiking")
incidents.group_activity=incidents.group_activity.replace("Snowshoeing and Hiking","Snowshoeing & Hiking")
incidents.group_activity=incidents.group_activity.replace("Mechanized Skiing","Heli or Cat Skiing")
incidents.group_activity=incidents.group_activity.replace("Heliskiing","Heli or Cat Skiing")
incidents.group_activity=incidents.group_activity.replace("At Outdoor Worksite","Work")
incidents.group_activity=incidents.group_activity.replace("Control Work","Work")
incidents.group_activity=incidents.group_activity.replace("Hunting/Fishing","Other Recreational")
incidents.group_activity=incidents.group_activity.replace("Inside Car/Truck on Road","Car/Truck/Building")
incidents.group_activity=incidents.group_activity.replace("Car/Truck on Road","Car/Truck/Building")
incidents.group_activity=incidents.group_activity.replace("Inside Building","Car/Truck/Building")
incidents.group_activity=incidents.group_activity.replace("Outside Building","Car/Truck/Building")


incidents.group_activity.unique()

fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,4))
colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
incidents.groupby(['group_activity']).id.count().plot(kind='bar', title="Incidents by Activity", ax=ax[0])
incidents.groupby(['group_activity']).num_fatal.sum().plot(kind='bar', title="Deaths by Activity", ax=ax[1], color=colors[1])
ax[0].set_xlabel(None)
ax[1].set_xlabel(None);
```

```{code-cell} python
incidents["date"] = pd.to_datetime(incidents.ob_date)
incidents["year"] = incidents.date.apply(lambda x: x.year)
incidents.date = incidents.date.dt.date
colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
f = incidents.groupby(["year"]).num_fatal.sum()
n = incidents.groupby(["year"]).id.count()
yearstart=1950
f=f[f.index>yearstart]
n=n[n.index>yearstart]
fig,ax = plt.subplots(1,1,figsize=(12,4))
n.plot(ax=ax)
f.plot(ax=ax)
ax.set_ylabel("Count")
ax.annotate("Incidents", (2010, 4), color=colors[0])
ax.annotate("Deaths", (2011, 15), color=colors[1]);
```

#### Mapping Incidents

Since the incident data includes coordinates, we might as well make a
map too. Unfortunately, some latitude and longitudes contain obvious errors.
Here, we try to fix them.

```{code-cell} python
import re

# fix errors in latitude, longitude
latlon = incidents.location_coords
def makenumeric(cstr):
    if cstr is None:
        return([None,None])
    elif (type(cstr)==str):
        return([float(s) for s in re.findall(r'-?\d+\.?\d*',cstr)])
    else:
        return(cstr)

latlon = latlon.apply(makenumeric)

def good_lat(lat):
    return(lat >= 41.6 and lat <= 83.12) # min & max for Canada

def good_lon(lon):
    return(lon >= -141 and lon<= -52.6)

def fixlatlon(c):
    if (len(c)<2 or type(c[0])!=float or type(c[1])!=float):
        c = [None, None]
        return(c)
    lat = c[0]
    lon = c[1]
    if not good_lat(lat) and good_lat(lon):
        tmp = lat
        lat = lon
        lon = tmp
    if not good_lon(lon) and good_lon(-lon):
        lon = -lon
    if not good_lon(lon) and good_lon(lat):
        tmp = lat
        lat = lon
        lon = tmp
    if not good_lon(lon) and good_lon(-lat):
        tmp = -lat
        lat = lon
        lon = tmp
    if not good_lat(lat) or not good_lon(lon):
        c[0] = None
        c[1] = None
    else:
        c[0] = lat
        c[1] = lon
    return(c)

incidents["latlon"] = latlon.apply(fixlatlon)
```

```{code-cell} python
def foo(c, a):
    if (type(a)==str):
        a = [float(s) for s in re.findall(r'-?\d+\.?\d*',a)]
    if len(a) <2:
        a = [None,None]
    return([a[1],a[0]] if type(c[0])!=float else c)
incidents["latlon_filled"]=[foo(c,a) for c,a in zip(incidents["latlon"],incidents["alt_coord"])]
nmiss = sum([a[0]==None for a in incidents.latlon_filled])
n = len(incidents.latlon_filled)
print("{} of {} incidents have latitude & longitude".format(n-nmiss, n))
```

```{code-cell} python
# download forecast region definitions
# req = urllib.request.Request("https://www.avalanche.ca/api/forecasts")
# The above link doesn't work since COVID-19 lockdown. Currently we use an old cached version instead
#req = ("https://web.archive.org/web/20150319031605if_/http://www.avalanche.ca/api/forecasts")
#with urllib.request.urlopen(req) as response:
#    forecastregions = json.loads(response.read().decode('utf-8'))
req = "https://faculty.arts.ubc.ca/pschrimpf/forecast-regions2015.json"
with urllib.request.urlopen(req) as response:
    regions2015 = json.loads(response.read().decode('utf-8'))

req = "https://faculty.arts.ubc.ca/pschrimpf/forecast-regions2019.json"
with urllib.request.urlopen(req) as response:
    regions2019 = json.loads(response.read().decode('utf-8'))

forecastregions = regions2019
ids = [r['id'] for r in forecastregions['features']]
for r in regions2015['features'] :
     if not r['id'] in ids :
            forecastregions['features'].append(r)
```

You may have to uncomment the second line below if  folium is not installed.

```{code-cell} python
# Map forecast regions and incidents
#!pip install --user folium
import folium
import matplotlib

cmap = matplotlib.colormaps["Set1"]
fmap = folium.Map(location=[60, -98],
                            zoom_start=3)
with urllib.request.urlopen(req) as response:
    regions_tmp = json.loads(response.read().decode('utf-8'))
folium.GeoJson(regions_tmp,
               tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=[""]),
               highlight_function=lambda x: { 'weight': 10},
              style_function=lambda x: {'weight':1}).add_to(fmap)
activities = incidents.group_activity.unique()
for i in incidents.index:
    if incidents.latlon_filled[i][0] is not None and  incidents.latlon_filled[i][1] is not None:
        cindex=[j for j,x in enumerate(activities) if x==incidents.group_activity[i]][0]
        txt = "{}, {}<br>{} deaths"
        txt = txt.format(incidents.group_activity[i],
                        incidents.ob_date[i],
                        incidents.num_fatal[i]
                        )
        pop = folium.Popup(incidents.comment[i], parse_html=True, max_width=400)
        folium.CircleMarker(incidents.latlon_filled[i],
                      tooltip=txt,
                      popup=pop,
                      color=matplotlib.colors.to_hex(cmap(cindex)), fill=True, radius=5).add_to(fmap)
fmap
```

Take a moment to click around the map and read about some of the incidents.

Between presenting this information on a map and the list on [https://www.avalanche.ca/incidents/](https://www.avalanche.ca/incidents/) ,
which do you prefer and why?

#### Matching Incidents to Regions

Later, we will want to match incidents to forecasts, so let's find the closest region to each incident.

Note that distance here will be in units of latitude, longitude (or
whatever coordinate system we use). At the equator, a distance of 1 is
approximately 60 nautical miles.

Since longitude lines get closer together farther from the equator,
these distances will be understated the further North you go.

This is not much of a problem if we're just finding the
nearest region, but if we care about accurate distances, we should
re-project the latitude and longitude into a different coordinate system.

```{code-cell} python
# Match incidents to nearest forecast regions.
from shapely.geometry import Point, Polygon, shape
point = Point(incidents.latlon_filled[0][1],incidents.latlon_filled[0][0])
def distances(latlon):
    point=Point(latlon[1],latlon[0])
    df = pd.DataFrame.from_dict([{'id':feature['id'],
                                  'distance':shape(feature['geometry']).distance(point)} for
                                 feature in forecastregions['features']])
    return(df)
def foo(x):
    if (x[0]==None):
        return(None)
    d = distances(x)
    return(d.id[d.distance.idxmin()])
incidents['nearest_region'] = incidents.latlon_filled.apply(foo)
incidents['nearest_distance'] = incidents.latlon_filled.apply(lambda x: None if x[0]==None else distances(x).distance.min())
```

```{code-cell} python
incidents
```

#### Forecast Data

We'll now download all forecasts for all regions since November 2011 (roughly the earliest data available).

We can only request one forecast at a time, so this takes many hours to download.

To make this process run more quickly for readers, we ran the code ourselves and then stored the data in the cloud.

The function below will fetch all the forecasts from the cloud storage location and save them to a folder
named `avalanche_forecasts`.

```{code-cell} python
def download_cached_forecasts():
    # download the zipped file and unzip it here
    url = "https://datascience.quantecon.org/assets/data/avalanche_forecasts.zip?raw=true"
    with requests.get(url) as res:
        if not res.ok:
            raise ValueError("failed to download the cached forecasts")
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            for f in z.namelist():
                if (os.path.isfile(f) and z.getinfo(f).file_size < os.stat(f).st_size):
                    warnings.warn(f"'File $f exists and is larger than version in cache. Not replacing.")
                else :
                    z.extract(f)

download_cached_forecasts()
```

The code below is what we initially ran to obtain all the forecasts.

You will notice that this code checks to see whether the files can be found in the `avalanche_forecasts`
directory (they can if you ran the `download_cached_forecasts` above!) and will only download them if they aren't found.

You can experiment with this caching by deleting one or more files from the `avalanche_forecasts`
folder and re-running the cells below.

```{code-cell} python
# Functions for downloading forecasts from Avalanche Canada

def get_forecast(date, region):
    url = "https://www.avalanche.ca/api/bulletin-archive/{}/{}.json".format(date.isoformat(),region)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
        return(result)
    except:
        return(None)

def get_forecasts(start, end, region):
    day = start
    forecasts = []
    while(day<=end and day<end.today()):
        #print("working on {}, {}".format(region,day))
        forecasts = forecasts + [get_forecast(day, region)]
        #print("sleeping")
        time.sleep(0.1) # to avoid too much load on Avalanche Canada servers
        day = day + pd.Timedelta(1,"D")
    return(forecasts)

def get_season(year, region):
    start_month = 11
    start_day = 20
    last_month = 5
    last_day = 1
    if (not os.path.isdir("avalanche_forecasts")):
        os.mkdir("avalanche_forecasts")
    seasonfile = "avalanche_forecasts/{}_{}-{}.json".format(region, year, year+1)
    if (not os.path.isfile(seasonfile)):
        startdate = pd.to_datetime("{}-{}-{} 12:00".format(year, start_month, start_day))
        lastdate = pd.to_datetime("{}-{}-{} 12:00".format(year+1, last_month, last_day))
        season = get_forecasts(startdate,lastdate,region)
        with open(seasonfile, 'w') as outfile:
            json.dump(season, outfile, ensure_ascii=False)
    else:
        with open(seasonfile, "rb") as json_data:
            season = json.load(json_data)
    return(season)
```

```{code-cell} python
forecastlist = [
    forecast
    for year in range(2011, 2019)
    for region in [region["id"] for region in forecastregions["features"]]
    for forecast in (get_season(year, region) or [])
    if forecast is not None
]
```

```{code-cell} python
# convert to DataFrame and extract some variables
forecasts = pd.DataFrame.from_dict([f for f in forecastlist if not f==None],orient="columns")

forecasts["danger_date"] = forecasts.dangerRatings.apply(lambda r: r[0]["date"])
forecasts["danger_date"] = pd.to_datetime(forecasts.danger_date, format='ISO8601').dt.date
forecasts["danger_alpine"]=forecasts.dangerRatings.apply(lambda r: r[0]["dangerRating"]["alp"])
forecasts["danger_treeline"]=forecasts.dangerRatings.apply(lambda r: r[0]["dangerRating"]["tln"])
forecasts["danger_belowtree"]=forecasts.dangerRatings.apply(lambda r: r[0]["dangerRating"]["btl"])
```

```{code-cell} python
forecasts.head()
```

```{code-cell} python
# merge incidents to forecasts
adf = pd.merge(forecasts, incidents, how="left",
               left_on=["region","danger_date"],
               right_on=["nearest_region","date"],
              indicator=True)
adf["incident"] = adf._merge=="both"
print("There were {} incidents matched with forecasts data. These occured on {}% of day-regions with forecasts".format(adf.incident.sum(),adf.incident.mean()*100))
```

```{code-cell} python
import seaborn as sns
ratings=sorted(adf.danger_alpine.unique())
ava_colors = ["#52BA4A", "#FFF300", "#F79218", "#EF1C29", "#1A1A1A", "#BFBFBF"]
for x in ["danger_alpine", "danger_treeline", "danger_belowtree"]:
    fig=sns.catplot(x=x, kind="count",col="incident", order=ratings, data=adf, sharey=False,
                    palette=ava_colors, height=3, aspect=2)
    plt.subplots_adjust(top=0.9)
    fig.fig.suptitle(x.replace("danger_",""))
    display(fig)
```

## Predicting Incidents from Text

### Preprocessing

The first step when using text as data is to pre-process the text.

In preprocessing, we will:

1. Clean: Remove unwanted punctuation and non-text characters.
1. Tokenize: Break sentences down into words.
1. Remove "stopwords": Eliminate common words that actually provide no information, like "a" and "the".
1. Lemmatize words: Reduce words to their dictionary "lemma" e.g. "snowing" and "snowed" both become snow (verb).

```{code-cell} python
from bs4 import BeautifulSoup
import nltk
import string
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Remove stopwords (the, a, is, etc)
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords=stopwords.union(set(string.punctuation))
# Lemmatize words e.g. snowed and snowing are both snow (verb)
wnl = nltk.WordNetLemmatizer()
def text_prep(txt):
    soup = BeautifulSoup(txt, "lxml")
    [s.extract() for s in soup('style')] # remove css
    txt=soup.text # remove html tags
    txt = txt.lower()
    tokens = [token for token in nltk.tokenize.word_tokenize(txt)]
    tokens = [token for token in tokens if not token in stopwords]
    #tokens = [token for token in tokens if not token ]
    tokens = [wnl.lemmatize(token) for token in tokens]
    if (len(tokens)==0):
        tokens = ["EMPTYSTRING"]
    return(tokens)

text_prep(forecasts.highlights[1000])
```

Now, let's apply this to all avalanche summaries.

```{code-cell} python
text_data = [text_prep(txt) for txt in adf.avalancheSummary]
```

Let's make a bar plot of the most common words.

```{code-cell} python
wf = nltk.FreqDist([word for doc in text_data for word in doc]).most_common(20)
words = [x[0] for x in wf]
cnt = [x[1] for x in wf]

fig, ax = plt.subplots(figsize=(12,4))
ax.bar(range(len(words)), cnt);
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
ax.set_title('Most common words in avalanche summaries');
ax.set_xlabel('Word');
ax.set_ylabel('Occurences');
plt.show()
```

### Feature Engineering

The "bag of words" approach is the simplest way to convert a collection of processed text
documents to a feature matrix. We view
each document as a bag of words, and our feature matrix
counts how many times each word appears. This method is called a "bag of words"
because we ignore the document's word order.

```{code-cell} python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7)
text_data = [text_prep(txt) for txt in adf.avalancheSummary]
y = adf.incident
X = vectorizer.fit_transform([' '.join(doc) for doc in text_data])
```

We can also perform more complicated feature engineering. One extension of the "bag of words" method
is to consider counts of pairs or triples of consecutive words. These
are called n-grams and can be created by setting the `n_gram`
argument to `CountVectorizer`. Another alternative might be to accommodate
the fact that common words will inherently have higher counts by using
term-frequency inverse-document-frequency (see below).

After creating our feature matrix, we can now apply any classification
method to predict incidents.

### Naive Bayes Classifier

A common text data classifier is the Naive Bayes classifier.
This classifier predicts incidents using Bayes' rules.

$$
P(incident | words) = \frac{P(words|incident) P(incidents)}{P(words)}
$$

The classifier is naive, though; it assumes words are independent of one another in any given incident.

$$
P(words|incident) = \prod_{w \in words} P(w|incident)
$$

Although this assumption is implausible for text, the Naive Bayes
classifier can be computed extremely quickly, and sometimes quite well.

```{code-cell} python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=124)
```

```{code-cell} python
from sklearn import naive_bayes
classifier = naive_bayes.MultinomialNB()
classifier.fit(Xtrain,ytrain)
np.mean(classifier.predict(Xtest)==ytest)
```

```{code-cell} python
from sklearn import metrics
print(metrics.confusion_matrix(ytest, classifier.predict(Xtest)))
print(metrics.classification_report(ytest, classifier.predict(Xtest)))
```

```{code-cell} python
# print text with highest predicted probabilities
phat=classifier.predict_proba(X)[:,1]
def remove_html(txt):
    soup = BeautifulSoup(txt, "lxml")
    [s.extract() for s in soup('style')] # remove css
    return(soup.text)
docs = [remove_html(txt) for txt in adf.avalancheSummary]
txt_high = [(_,x) for _, x in sorted(zip(phat,docs), key=lambda pair: pair[0],reverse=True)]
txt_high[:10]
```

```{code-cell} python
# print text with lowest predicted probabilities
txt_low = [(_,x) for _, x in sorted(zip(phat,docs), key=lambda pair: pair[0])]
txt_low[:10]
```

````{admonition} Exercise
:name: app-txt-dir1

See exercise 1 in the {ref}`exercise list <app-txt-ex>`.
````

Predicting deaths from forecast text is very difficult because deaths
are so rare. A prediction exercise more likely to succeed would be to
predict the avalanche rating from the forecast text. However,
doing so is a very
artificial task, with little practical use.

Another alternative would be to gather more data on non-fatal
avalanches. Avalanche Canada also has user-submitted "Mountain
Information Network" reports. These reports include observations of
natural avalanches and information on non-fatal avalanche
incidents. Since the data is user-submitted, it is messy and more
difficult to work with. Nonetheless, working with it would be
good practice and could lead to some insights.

## Unsupervised Learning

The regression and classification methods that we have seen so far are
examples of supervised learning --- we are trying to predict an observed outcome.
In unsupervised learning, we do not have an
observed outcome to predict. Instead, we try to find informative
patterns in the data. Unsupervised learning can be particularly useful
with text data. We will look at two related techniques for topic
modeling. These techniques attempt to extract distinct topics from a
collection of text documents.

### Latent Semantic Analysis

Latent semantic analysis is used by some search engines to rank
the similarities among documents. Latent semantic analysis begins with a
term document matrix, $X$. The term document matrix is a number
of documents by number of terms matrix where the i,jth entry is the
measure of how often term j appears in document i. This could be the
same bag of words feature matrix we constructed above, or it could be
some other measure. For this example, we will use the term-frequency,
inverse-document-frequency representation.

$$
x^{tfidf}_{ij} = \frac{\text{occurences of term j in document
i}}{\text{length of document i}} \log \left(\frac{\text{number of
documents}}{\text{number of documents containing term j}}\right)
$$

Given a term document matrix, $X$, latent semantic analysis
computes a lower rank approximation to $X$ through the singular
value decomposition. This lower rank approximation can potentially be
interpreted or used instead of $X$ for other learning
algorithms. In other contexts, similar decompositions are referred to
as principal components analysis or factor models.

```{code-cell} python
# LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
X = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in text_data])
```

```{code-cell} python
svd_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
```

Here, we have computed a rank 10 approximation to the tf-idf matrix. We
can see how much variance of the original matrix that our 10
components reproduce. We can also look at how all terms in the
document contribute to each of the 10 components.

```{code-cell} python
print(svd_model.explained_variance_ratio_)
print(svd_model.explained_variance_ratio_.cumsum())
terms = tfidf_vectorizer.get_feature_names_out() 
comp_label=[]
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    message = ""
    for t in sorted_terms:
        message = message + "{:.2f} * {} + ".format(t[1],t[0])
    print(message)
    comp_label.append(message)
```

Finally, we can attempt to visualize the components.

The LSA reduced the dimensionality of the representation of documents
from thousands (of term-frequency inverse-document-frequency) to ten
components. While ten is more manageable than thousands, it is still too
many dimensions to effectively visualize. t-SNE is a technique to further
reduce dimensionality. t-SNE is a nonlinear data transformation
from many dimensions to 2, while attempting to preserve the
original clustering of their original domain. In other words, if a
set of documents are closely clustered in the 10 dimensional
LSA space, then they will also be close together in the 2 dimensional
t-SNE representation.

```{code-cell} python
lsa_topic_matrix = svd_model.transform(X)
```

```{code-cell} python
from sklearn.manifold import TSNE
nplot = 2000 # reduce the size of the data to speed computation and make the plot less cluttered
lsa_topic_sample = lsa_topic_matrix[np.random.choice(lsa_topic_matrix.shape[0], nplot, replace=False)]
tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=500,
                      n_iter=1000, verbose=10, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_sample)
```

The t-SNE model creates a
non-linear projection from our 10 dimensional LSA topics onto two dimensional space.
It can be useful for visualizing high-dimensional data. One word of caution:
the output of the t-SNE model can depend on the parameters of the
algorithm. Failure to see clear clusters in the t-SNE visualization
could mean either the original data was not clustered in higher
dimensional space or that the t-SNE algorithm parameters were
chosen poorly.

```{code-cell} python
cmap = matplotlib.colormaps["Paired"]
fig, ax = plt.subplots(1,2,figsize=(16,6))
n_topics=len(svd_model.components_)
lsa_keys = np.argmax(lsa_topic_sample, axis=1)
ax[0].scatter(x=tsne_lsa_vectors[:,0],y=tsne_lsa_vectors[:,1], color=[cmap(i) for i in lsa_keys], alpha=0.8)
bbox_props = dict(boxstyle="round4,pad=0.1", lw=0.2, fc="white")
for i in range(n_topics):
    m = tsne_lsa_vectors[lsa_keys==i, :].mean(axis=0)
    ax[0].text(m[0], m[1], str(i), ha="center", va="center",
               size=15, color=cmap(i),
               bbox=bbox_props)
    ax[1].text(0,1-(i+1)*1/(n_topics+1),"Topic " + str(i) + " : "+ comp_label[i],ha="left", va="center", color=cmap(i))
    ax[1].axis('off')
fig.tight_layout()
```

From this plot, we can immediately see two things. First, most documents
are closest to topic 0. Second, most topics are not
well-separated.

````{admonition} Exercise
:name: app-txt-dir2

See exercise 2 in the {ref}`exercise list <app-txt-ex>`.
````

### Latent Dirichlet Analysis

Latent dirichlet analysis (LDA) produces similar outputs as latent semantic
analysis, but LDA often produces nicer results. The statistical theory
underlying LSA is built on continuous $X$ features. LDA uses
similar ideas, but takes into account that text is discrete.

```{code-cell} python
# LDA
import gensim
# gensim works with a list of lists of tokens
text_data = [text_prep(txt) for txt in forecasts.avalancheSummary]
```

```{code-cell} python
# convert to bag of words
dictionary = gensim.corpora.Dictionary(text_data)
bow_data = [dictionary.doc2bow(text) for text in text_data]
```

```{code-cell} python
ldamodel = gensim.models.ldamodel.LdaModel(bow_data, num_topics = 5, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
```

```{code-cell} python
import pyLDAvis
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()
lda_display = pyLDAvis.gensim_models.prepare(ldamodel, bow_data, dictionary)
lda_display
```

````{admonition} Exercise
:name: app-txt-dir3

See exercise 3 in the {ref}`exercise list <app-txt-ex>`.
````

````{admonition} Exercise
:name: app-txt-dir4

See exercise 4 in the {ref}`exercise list <app-txt-ex>`.
````

(app-txt-ex)=
## Exercises

### Exercise 1

Use another classification method to predict incidents. Check whether
your method outperforms the Naive Bayes classifier.

({ref}`back to text <app-txt-dir1>`)


### Exercise 2

Apply LSA to the weather or snowpack descriptions. Can you notice
any patterns?

({ref}`back to text <app-txt-dir2>`)

### Exercise 3

Apply LDA to the weather or snowpack descriptions. Can you notice
any patterns?

({ref}`back to text <app-txt-dir3>`)


### Exercise 4

Use the reduced rank representation of text from LSA or LDA as a
feature matrix to predict avalanche incidents. Compare the
performance with the bag of words feature matrix.

({ref}`back to text <app-txt-dir4>`)
