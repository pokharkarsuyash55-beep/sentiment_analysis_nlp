# Deployment


import streamlit as st
import pandas as pd
import pickle
import re

# NLP LIBRARIES
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==================================
# Load the model and vectorizer
# ==================================

model=pickle.load(open('model.pkl', 'rb'))

vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))

# ==================================
#  STOP WORDS + LEMMATIZATION
# ==================================

stopwords=set(stopwords.words('english'))

lemmatizer=WordNetLemmatizer()

# ==================================
# clean function
# ==================================

def clean_text(text):
    text=str(text).lower()
    # Remove URLs
    text=re.sub(r'http\S+','',text)

    # remove HTML
    text=re.sub(r'<.*?>','',text)

    # remove special characters 
    text=re.sub(r'[^a-zA-Z]',' ',text)

    # tokenization
    words=text.split()

    # remove small words
    words=[w for w in words if len(w)>2]

    # remove stop words
    words=[w for w in words if w not in stopwords]

    # lemmatization
    words=[lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ==================================
# Title
# ==================================
st.title("NLP SENTIMENT ANALYSIS")

# ==================================
# mode
# =================================

mode=st.radio(
    'choose option',
    ['Input Text','Upload File']
)

# ==================================
# Input Text mode
# ==================================

if mode=='Input Text':
    text=st.text_area("Enter Review ")

    if st.button("Predict"):
        # clean the text
        clean=clean_text(text)
        # vectorize the text
        vector=vectorizer.transform([clean])
        # predict the sentiment 
        prediction=model.predict(vector)[0]
        
        st.success(f"Prediction: {prediction}")

# ==================================
# Upload File mode
# ==================================

else:
    file=st.file_uploader(
        "Upload CSV or EXCEL file", 
        type=['csv','xlsx']
    )
    
    if file is not None:
        # ============================
        # read the file
        # ============================
        if file.name.endswith('.csv'):
            df=pd.read_csv(file)
        else:
            df=pd.read_excel(file)

        # ============================
        # show original data
        # ============================

        st.subheader('Original Data')
        st.write(df.head())

        # ============================
        # create review column
        # ============================

        if 'title' in df.columns and 'body' in df.columns:
            df['review']=(
                df['title'].fillna('').astype(str) + 
                ': ' + df['body'].fillna('').astype(str)
            )
        elif 'review' in df.columns:
            df['review']=df['review'].astype(str)
        else:
            st.error(
                "File must contain review column OR title/body columns"
            )
            st.stop()

        # ============================
        # process file button
        # ============================
        if st.button('Process file'):
            # remove empty reviews
            df=df[
                df['review'].str.strip() != ""
            ]

            # remove duplicates
            df.drop_duplicates(
                subset='review',
                inplace=True
            )

            # remove non english reviews
            df=df[
                df['review'].str.contains(
                    r'[a-zA-Z]',
                    regex=True
                )
            ]

            # ============================
            # text cleaning
            # ============================

            df['clean_review']=df['review'].apply(clean_text)

            # ============================
            # TF-IDF vectorization
            # ============================
            vec=vectorizer.transform(
                df['clean_review']
            )

            # ============================
            # prediction
            # ============================

            df['prediction']=model.predict(vec)

            # ============================
            # show results
            # ============================

            st.subheader('Prediction Result')

            st.write(
                df[
                    ['review','prediction']
                ].head()
            )

            # ============================
            # sentiment count
            # ============================

            st.subheader('Sentiment Count')

            st.write(
                df['prediction'].value_counts()
            )

            # ============================
            # sentiment distribution
            # ============================
            st.subheader('Sentiment Distribution')
            st.bar_chart(
                df['prediction'].value_counts()
            )

            # ============================
            # download results
            # ============================

            csv=df.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                'Download Results',
                csv,
                'sentiment_analysis_results.csv',
                'text/csv'
            )