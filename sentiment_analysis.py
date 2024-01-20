import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
import seaborn as sns
from livelossplot import PlotLossesKeras

# Load the data
data = pd.read_csv('tripadvisor_hotel_reviews.csv')

rating_counts = data['Rating'].value_counts()
df_counts = pd.DataFrame({'rating': rating_counts.index, 'count': rating_counts.values})

sns.set_theme(style="whitegrid")
sns.barplot(x="rating", y="count", data=df_counts)
plt.title('TripAdvisor hotel reviews Ratings count')
plt.show()

# Remove punctuations from the text data
data['Review'] = data['Review'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Remove digits from the text data
data['Review'] = data['Review'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))

# Convert text data into lower case
data['Review'] = data['Review'].apply(lambda x: x.lower())

# Remove stop words from the text data
stop_words = set(stopwords.words('english'))
data['Review'] = data['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Perform stemming on the text data
stemmer = PorterStemmer()
data['Review'] = data['Review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Separate the data into features and target
X = data['Review']
y = data['Rating']

# Define a function to convert ratings to sentiment categories
def convert_to_sentiment(rating):
    if rating == 3:
        return 1.0 #Neutral
    elif rating > 3:
        return 2.0 #positive
    else:
        return 0.0 #negative

# Convert ratings to sentiment categories
y = y.apply(convert_to_sentiment)

# Convert text data into bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bigram = vectorizer.fit_transform(X)

# Convert text data into word embeddings
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_embed = tokenizer.texts_to_sequences(X)
X_embed = pad_sequences(X_embed, maxlen=100)

#feature selection
k = 1000
selector = SelectKBest(mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_bigram, y)

X_concat = hstack([X_selected, X_embed])
X_concat = X_concat.toarray()
X_concat = X_concat[:, :100]


def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True)

    return model, es

# Perform K-Fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)

cvscores = []

for train, test in kfold.split(X_concat, y):
    # Split the data into training and testing sets
    X_train, X_test = X_concat[train], X_concat[test]
    y_train, y_test = y[train], y[test]

    # Encode the target variable
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Train the model
    model, es = create_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0, validation_data=(X_test, y_test),
                        callbacks=[es])
    print(model.summary())

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print("the accuracy score:", acc)
    cr = classification_report(y_test, y_pred, zero_division=0)
    print(cr)

    cvscores.append(acc)

# Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
    print(cm)
    # Plot the confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Print the average accuracy score
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores) * 100, np.std(cvscores) * 100))

# Collect the loss and accuracy values during training
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the loss and accuracy values
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Train the model on the full dataset
model, es = create_model()
model.fit(X_concat, y, epochs=10, batch_size=64, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=3)])

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Make predictions on new data
new_data = ["Excellent hotel; I like the sea-facing balcony.", "good staff,price was high but affordable"]
new_data_bigram = vectorizer.transform(new_data)
new_data_selected = selector.transform(new_data_bigram)
new_data_embed = tokenizer.texts_to_sequences(new_data)
new_data_embed = pad_sequences(new_data_embed, maxlen=100)
new_data_concat = hstack([new_data_selected, new_data_embed])
new_data_concat = new_data_concat.toarray()
new_data_concat = new_data_concat[:, :100]
new_data_pred_prob = model.predict(new_data_concat)
new_data_pred = np.argmax(new_data_pred_prob, axis=1)
new_data_pred = le.inverse_transform(new_data_pred)
print("prediction:",new_data_pred)






