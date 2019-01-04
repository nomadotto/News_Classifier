
# doing basic imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, metrics
import numpy as np
import functools
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
import itertools

# let's read in the data and take a gander at it
# it's pre-cleaned (by visual inspection) so I don't have to do a not of pre-processing
data = pd.read_json('./News_Category_Dataset_v2.json', lines=True)
print(data.head())
print(data.columns)

# let's see the categories
data.category.unique()

# get label counts
n_records = data.category.value_counts()
print(n_records)


# do basic Plotting
fig, ax = plt.subplots()
fig.set_size_inches(8, 11)
ind = range(len(n_records))
width = 0.35
bars = ax.barh(ind, n_records.values, width)
labels = n_records.index.tolist()
ax.set_yticks(ticks=ind)
ax.set_yticklabels(labels)
ax.set_xlabel('Number of Records in the News Category Dataset with a Given Category')
ax.set_title('Number of News Stories by Category ')
plt.show()

# now to start building a simple classifier
# I'm going to start with using term frequency in headline and short_decription (smooshing them together)
# as our features because it's relatively straightforward. There's likely a lot of improvement to be made by
# considering them as separate features rather than a single blob, but performance isn't really the goal
# going to use the ol' TfidfVectorizer in sklearn to do the vectorization


# first I'm going to make sure that it works ok
blob_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
# including stopwords to keep size under control (ish)
blob_transformer = blob_vectorizer.fit(data.headline + ' ' + data.short_description)
transformed = blob_transformer.transform(data.headline + ' ' + data.short_description)

# yup, it goes. We've got some weird words (see headline_transformer.get_feature_names()
# or headline_transformer.vocabulary_, so I'm gonna double-check that there's no nonsense happening

for elem in data.headline:
    weirdos = ['ça', 'édgar', 'élysées', 'émigrés', 'émilion', 'étienne', 'étretat', 'über', 'čechomor', 'ένας',
               'αστυνομίας', 'διαδηλωτών', 'κάιρο', 'και', 'μεταξύ', 'νεκρός', 'σε', 'στο', 'συγκρούσεις', 'ᵒᴥᵒᶅ']
    if any(oddball in elem for oddball in weirdos):
        print(elem)
#
# ok, I'm convinced (sorta). Looks like we have some headlines which are not in english?
# I think they're gonna cause some problems with the classifier, since they look rare.
# I can drop them, which is probably the best move, so I'll do that.

greek_words = ['ένας', 'αστυνομίας', 'διαδηλωτών', 'κάιρο', 'και', 'μεταξύ', 'νεκρός', 'σε', 'στο', 'συγκρούσεις']
cleaned_data = data[~(data['headline'] + ' ' + data['short_description']).str.contains('|'.join(greek_words))]
cleaned_data = cleaned_data.join((cleaned_data['headline']
                                  + ' ' + cleaned_data['short_description']).rename('text_blob'))


x_cols = ['text_blob']
# the second feature I'm considering is a categorical on author
# I see there's a bunch of 'contributors', so I'll add that as another feature
cleaned_data = cleaned_data.join(cleaned_data.authors.str.contains('Contributor').rename('is_contributor'))

# there's a lot of authors
print(len(cleaned_data.authors.str.strip('\n').unique()))
# let's look at the spread
n_auth = data.authors.str.strip('\n').value_counts()
print(n_auth)


# gonna do it for a subset of authors who write a lot
# small authors will be our "cold" in the 1-hot scheme
authors_to_dummy = n_auth[n_auth > 100].index

for author in authors_to_dummy:
    author = author.replace('(', '')
    author = author.replace(')', '')
    author = author.replace('\n', ' ')
    author = author.replace(' ', '_')  # whitespace is messy
    # try/except to make sure authors with issues get handled, because text data is gross
    try:
        name = 'is_' + author[0:30]
        if name in cleaned_data.columns:
            name += '_2'
        cleaned_data = cleaned_data.join(cleaned_data.authors.str.contains(author).rename(name), rsuffix='2')
        x_cols.append(name)
    except Exception as e:
        print(e)
        print(author)

# another feature that might be useful is a time-based guy. I'm going to build a set of fourier coeffs
# and work off those. Let's start off looking at the timestamp data
print(cleaned_data.date.describe())
print(cleaned_data.date.value_counts().head())

# looks like we have ~100 stories each day. So we're going to have a lot of information in the feature to work with

cleaned_data['month'] = cleaned_data.date.dt.month
cleaned_data['year'] = cleaned_data.date.dt.year
cleaned_data['day_in_month'] = cleaned_data.date.dt.day


def make_fourier_value(x, max_val, min_val, sin_cos='sin', k=1):
    if sin_cos.lower().strip() == 'sin':
        val = np.sin(2 * x * np.pi * k / (max_val - min_val +1 ))
    elif sin_cos.lower().strip() == 'cos':
        val = np.sin(2 * x * np.pi * k / (max_val - min_val +1 ))
    else:
        val = np.nan
    return val


for n in range(1, 10):
    for half in ['sin', 'cos']:
        for col in ['month', 'year', 'day_in_month']:
            maximum = cleaned_data[col].max()
            minimum = cleaned_data[col].min()
            fourier = functools.partial(make_fourier_value, max_val=maximum, min_val=minimum, sin_cos=half, k=n)
            name = col + '_' + half + '_' + str(n)
            cleaned_data = cleaned_data.join(cleaned_data[col].apply(fourier).rename(name))
            x_cols.append(name)


# Now I'm gonna to do the fitting. I'm gonna start by doing a train/test split

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(cleaned_data[x_cols],
                                                                      cleaned_data['category'],
                                                                      stratify=cleaned_data['category'])
# let's use that blob vectorizer to build the features

blob_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
blob_transformer = blob_vectorizer.fit(train_x['text_blob'])
transformed_train = blob_transformer.transform(train_x['text_blob'])
transformed_val = blob_transformer.transform(valid_x['text_blob'])


# now to throw the features together and build a model
x_cols.remove('text_blob')  # we want to remove the text data now that it's been transformed, since we can't fit with it
X_train_final = hstack((transformed_train, csr_matrix(train_x[x_cols].values.astype(float))))
X_valid_final = hstack((transformed_val, np.array(valid_x[x_cols].values.astype(float))))


params = {'n_estimators': 100, 'max_depth': 500}  # setting some reasonable parameters
RFC = RandomForestClassifier(**params)  # Using a random forest, because I like them
RFC.fit(X_train_final, train_y)
insample_score = RFC.score(X_train_final, train_y)  # getting the score
insample_predict = RFC.predict(X_train_final)
print(insample_score)  # looks good, suspiciously good. We probably over-trained.
oos_score = RFC.score(X_valid_final, valid_y)
oos_predict = RFC.predict(X_valid_final)
print(oos_score)  # looks like we've over-trained, I'm not going to sweat it too much for this example, but would ...
# ...need to be fixed for any kind of production

# Let's look at the confusion matrices to see what we can see
labels = cleaned_data.category.unique().astype(list)
insample_labels = train_y.unique().astype(list)
oos_labels = valid_y.unique().astype(list)


insample_confusion_matrix = metrics.confusion_matrix(train_y, insample_predict, labels=insample_labels)
oos_confusion_matrix = metrics.confusion_matrix(valid_y, oos_predict, labels=oos_labels)

plt.figure()

# borrowing from https://scikit-learn.org/stable/auto_examples/...
# ...model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# to do pretty plotting of confusion matricies


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=7)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# doing some plotting
np.set_printoptions(precision=2)
plt.figure(figsize=(13, 13))
plot_confusion_matrix(insample_confusion_matrix, classes=insample_labels, normalize=True,
                      title='Normalized In-Sample Confusion Matrix')
plt.show()

plt.figure(figsize=(13, 13))

plot_confusion_matrix(oos_confusion_matrix, classes=oos_labels, normalize=True,
                      title='Normalized Out-of-Sample Confusion Matrix')
plt.show()

# yeah, looks like we have overfitting on popular categories. We need to try some weighting if we want to
# improve performance.


params = {'n_estimators': 100, 'class_weight':'balanced_subsample',
          'min_samples_split': 10}  # removed the depth restriction,
#  which might increase overfitting, instead requiring min-samples to allow it to take best advantage of
# the number of features
RFC = RandomForestClassifier(**params)  # Using a random forest, because I like them
RFC.fit(X_train_final, train_y)
insample_score = RFC.score(X_train_final, train_y)  # getting the score
insample_predict = RFC.predict(X_train_final)
print(insample_score)
oos_score = RFC.score(X_valid_final, valid_y)
oos_predict = RFC.predict(X_valid_final)
print(oos_score)

insample_confusion_matrix = metrics.confusion_matrix(train_y, insample_predict, labels=insample_labels)
oos_confusion_matrix = metrics.confusion_matrix(valid_y, oos_predict, labels=oos_labels)

plt.figure(figsize=(13, 13))
plot_confusion_matrix(insample_confusion_matrix, classes=insample_labels, normalize=True,
                      title='Normalized In-Sample Confusion Matrix')
plt.show()

plt.figure(figsize=(13, 13))

plot_confusion_matrix(oos_confusion_matrix, classes=oos_labels, normalize=True,
                      title='Normalized Out-of-Sample Confusion Matrix')
plt.show()

# score went down. I don't notice a big gain in accuracy, which is disappointing, but expected.
#