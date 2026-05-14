# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import pickle
import re
import nltk

# DOWNLOAD NLTK DATA
nltk.download('stopwords')
nltk.download('wordnet')

# NLP LIBRARIES
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# LOAD MODEL + VECTORIZER
# ==========================================

# LOAD TRAINED MODEL
model = pickle.load(
    open('model.pkl','rb')
)

# LOAD TF-IDF VECTORIZER
vectorizer = pickle.load(
    open('vectorizer.pkl','rb')
)

# ==========================================
# STOPWORDS + LEMMATIZER
# ==========================================

# REMOVE COMMON WORDS
stop_words = set(
    stopwords.words('english')
)

# ROOT WORD CONVERSION
lemmatizer = WordNetLemmatizer()

# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    # CONVERT TO LOWERCASE
    text = str(text).lower()

    # REMOVE URL
    text = re.sub(
        r'http\S+',
        '',
        text
    )

    # REMOVE HTML TAGS
    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    # REMOVE SPECIAL CHARACTERS
    text = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    # TOKENIZATION
    words = text.split()

    # REMOVE SMALL WORDS
    words = [
        w for w in words
        if len(w) > 2
    ]

    # REMOVE STOPWORDS
    words = [
        w for w in words
        if w not in stop_words
    ]

    # LEMMATIZATION
    words = [
        lemmatizer.lemmatize(w)
        for w in words
    ]

    # RETURN CLEAN TEXT
    return " ".join(words)

# ==========================================
# STREAMLIT TITLE
# ==========================================

st.title(
    "NLP SENTIMENT ANALYSIS SYSTEM"
)

# ==========================================
# SELECT MODE
# ==========================================

mode = st.radio(

    "Choose Option",

    ["Input Text","Upload File"]

)

# ==========================================
# INPUT TEXT MODE
# ==========================================

if mode == "Input Text":

    # USER INPUT
    text = st.text_area(
        "Enter Review"
    )

    # PREDICT BUTTON
    if st.button("Predict"):

        # CLEAN TEXT
        clean = clean_text(text)

        # TF-IDF VECTORIZATION
        vector = vectorizer.transform(
            [clean]
        )

        # MODEL PREDICTION
        prediction = model.predict(
            vector
        )[0]

        # SHOW RESULT
        st.success(
            f"Prediction : {prediction}"
        )

# ==========================================
# FILE UPLOAD MODE
# ==========================================

else:

    # FILE UPLOADER
    file = st.file_uploader(

        "Upload CSV or Excel File",

        type=['csv','xlsx']

    )

    # CHECK FILE
    if file is not None:

        # ==================================
        # READ FILE
        # ==================================

        if file.name.endswith('.csv'):

            df = pd.read_csv(file)

        else:

            df = pd.read_excel(file)

        # ==================================
        # SHOW ORIGINAL DATA
        # ==================================

        st.subheader(
            "Original Data"
        )

        st.write(
            df.head()
        )

        # ==================================
        # CREATE REVIEW COLUMN
        # ==================================

        if 'title' in df.columns and 'body' in df.columns:

            df['review'] = (

                df['title']
                .fillna('')
                .astype(str)

                + ' : ' +

                df['body']
                .fillna('')
                .astype(str)

            )

        elif 'review' in df.columns:

            df['review'] = df['review'].astype(str)

        else:

            st.error(

                "File must contain review column OR title/body columns"

            )

            st.stop()

        # ==================================
        # PROCESS FILE BUTTON
        # ==================================

        if st.button("Process File"):

            # REMOVE EMPTY REVIEWS
            df = df[
                df['review'].str.strip() != ""
            ]

            # REMOVE DUPLICATES
            df.drop_duplicates(

                subset='review',

                inplace=True

            )

            # REMOVE NON ENGLISH TEXT
            df = df[
                df['review'].str.contains(
                    r'[a-zA-Z]',
                    regex=True
                )
            ]

            # ==================================
            # TEXT CLEANING
            # ==================================

            df['clean_review'] = df['review'].apply(
                clean_text
            )

            # ==================================
            # REVIEW LENGTH
            # ==================================

            df['review_length'] = df['review'].apply(
                lambda x: len(x.split())
            )

            # ==================================
            # SHOW CLEANED DATA
            # ==================================

            st.subheader(
                "Cleaned Data"
            )

            st.write(

                df[
                    ['review','clean_review']
                ].head()

            )

            # ==================================
            # TF-IDF VECTORIZATION
            # ==================================

            vector = vectorizer.transform(

                df['clean_review']

            )

            # ==================================
            # PREDICTION
            # ==================================

            df['prediction'] = model.predict(
                vector
            )

            # ==================================
            # SHOW RESULT
            # ==================================

            st.subheader(
                "Prediction Result"
            )

            st.write(

                df[
                    ['review','prediction']
                ].head()

            )

            # ==================================
            # SENTIMENT COUNT
            # ==================================

            st.subheader(
                "Sentiment Count"
            )

            st.write(

                df['prediction'].value_counts()

            )

            # ==================================
            # BAR CHART
            # ==================================

            st.subheader(
                "Sentiment Distribution"
            )

            st.bar_chart(

                df['prediction'].value_counts()

            )

            # ==================================
            # DOWNLOAD RESULT
            # ==================================

            csv = df.to_csv(

                index=False

            ).encode('utf-8')

            st.download_button(

                "Download Result",

                csv,

                "sentiment_prediction.csv",

                "text/csv"

            )