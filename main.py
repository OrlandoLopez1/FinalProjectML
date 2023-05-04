import json
import pandas as pd
from PIL import Image
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.DataFrame(columns=['bussiness_id', 'reviews', 'photos', 'city', 'state', 'stars'])

business_ids = set()
states = []
cities = []
stars = []

with open('yelp_academic_dataset_business.json', encoding='utf-8') as f:
    count = 0
    for line in f:
        json_object = json.loads(line)
        business_ids.add(json_object.get('business_id'))
        states.append(json_object.get('state'))
        cities.append(json_object.get('city'))
        stars.append(json_object.get('stars'))
        count += 1
        if count == 1000:
            break

d = defaultdict(list)
reviews = []
c = 0
with open('yelp_academic_dataset_review.json', encoding='utf-8') as f1:
    for line in f1:
        review = json.loads(line)
        if review['business_id'] in business_ids:
            c += 1
            print(c)
            # reviews.append(review.get())
            d[review['business_id']].append(review.get('text'))
        if c == 50857:
            break

photos = []
c = 0
d_photos = defaultdict(list)
with open('photos.json', encoding='utf-8') as f2:
    for line in f2:
        photo = json.loads(line)
        if photo['business_id'] in business_ids:
            c += 1
            print(c)
            id = photo.get('photo_id')
            d_photos[photo['business_id']].append(f'photos/{id}.jpg')

photos_df = pd.DataFrame(list(d_photos.items()), columns=['business_id', 'photos'])
reviews_df = pd.DataFrame(list(d.items()), columns=['business_id', 'reviews'])
location_df = pd.DataFrame({'business_id': list(business_ids), 'state': states, 'city': cities, 'stars': stars})
merged_df = pd.merge(pd.merge(photos_df, reviews_df, on='business_id'), location_df, on='business_id')


def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    text = ' '.join(words)

    return text


def resize_image(image, size=(64, 64)):
    return image.resize(size, Image.LANCZOS)


def is_valid_image(image_path):
    try:
        Image.open(image_path).verify()
        return True
    except Exception as e:
        print(f"Invalid image: {image_path} - Error: {e}")
        return False


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize the pixel values
    return image_array


def compute_average_photo(image_paths):
    if not image_paths:
        return np.zeros((64, 64, 3))

    images = [np.array(resize_image(Image.open(path))) for path in image_paths if is_valid_image(path)]
    return np.mean(images, axis=0)


def concatenate_reviews(reviews):
    return ' '.join(reviews)


merged_df['concatenated_reviews'] = merged_df['reviews'].apply(concatenate_reviews)
merged_df['concatenated_reviews'] = merged_df['concatenated_reviews'].apply(preprocess_text)
merged_df['average_photo'] = merged_df['photos'].apply(compute_average_photo)

train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

X_train_images = np.stack(train_df['average_photo'].values)
X_test_images = np.stack(test_df['average_photo'].values)

X_train_images_flat = X_train_images.reshape(X_train_images.shape[0], -1) / 255.0
X_test_images_flat = X_test_images.reshape(X_test_images.shape[0], -1) / 255.0

n_components = 100
pca = PCA(n_components=n_components)
X_train_img_features = pca.fit_transform(X_train_images_flat)
X_test_img_features = pca.transform(X_test_images_flat)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(train_df['concatenated_reviews']).toarray()
X_test_tfidf = vectorizer.transform(test_df['concatenated_reviews']).toarray()

X_train_combined = np.concatenate((X_train_tfidf, X_train_img_features), axis=1)
X_test_combined = np.concatenate((X_test_tfidf, X_test_img_features), axis=1)

y_train_cat = pd.Categorical(train_df['stars'], categories=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y_test_cat = pd.Categorical(test_df['stars'], categories=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

y_train_codes = y_train_cat.codes
y_test_codes = y_test_cat.codes


models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train_combined, y_train_codes)
    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score(y_test_codes, y_pred)
    print(f'Model:{name}')
    print(classification_report(y_test_codes, y_pred, zero_division=0))
