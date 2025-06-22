import json
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load FAQ data
with open('faqs.json', 'r') as file:
    data = json.load(file)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words and word not in string.punctuation])

# process all questions
clean_questions = [clean_text(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_questions)

def get_answer(user_input):
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    similarities = cosine_similarity(user_vec, X)
    index = np.argmax(similarities)
    return answers[index]

# Chat loop
print("🤖 FAQ Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye! 👋")
        break
        
    response = get_answer(user_input)
    print("Bot:", response)
