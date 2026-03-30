# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import streamlit as st
#
#
# # Load the file into a DataFrame
# data = pd.read_csv('spam.csv')
#
# data.drop_duplicates(inplace=True)
# data['Category'] = data['Category'].replace(['ham', 'spam'],['Not Spam','Spam'])
#
# mess = data['Message']
# cat = data['Category']
#
# (mess_train, mess_test,cat_train,cat_test)= train_test_split(mess,cat,test_size=0.2)
#
# cv = CountVectorizer(stop_words='english')
# features = cv.fit_transform(mess_train)
#
# #creating model
#
# model = MultinomialNB()
# model.fit(features,cat_train)
#
# #test model;
# features_test = cv.transform(mess_test)
#
# #predict data
# def predict(message):
#     input_message = cv.transform([message]).toarray()
#     result = model.predict(input_message)
#     return result
#
# st.header('Spam Detector')
#
# input_mess =st.text_input('Enter your message Here')
#
# if st.button('Predict'):
#    output = predict(input_mess)
#    st.markdown(output)

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


# 1. Load data
data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)

# 2. FIX: Explicitly map labels to numbers to avoid confusion
# ham (Not Spam) -> 0, spam (Spam) -> 1
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

mess = data['Message']
cat = data['Category']

# Split data
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# 3. Vectorize
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# 4. Train Model
model = MultinomialNB()
model.fit(features, cat_train)

# 5. Predict function with fixed output mapping
def predict(message):
    input_message = cv.transform([message]) # No need for .toarray() here
    result = model.predict(input_message)
    # Convert the numeric prediction back to text
    return "Spam" if result[0] == 1 else "Not Spam"

# Streamlit UI
st.header('Spam Detector')
input_mess = st.text_input('Enter your message Here')

if st.button('Predict'):
    if input_mess:
        output = predict(input_mess)
        # 6. UI Improvement: Show colors based on result
        if output == "Spam":
            st.error(f"Prediction: {output}")
        else:
            st.success(f"Prediction: {output}")
    else:
        st.warning("Please enter a message first.")
