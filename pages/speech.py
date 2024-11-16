# pip install SpeechRecognition
# pip install pyaudio
import streamlit as st
import speech_recognition as sr
import nltk
# nltk.download()
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_lemmatize(sentence):
    doc = nlp(sentence)
    return " ".join([token.lemma_ for token in doc])

# lemmatizer = WordNetLemmatizer()
def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(spacy_lemmatize(word))
  return " ".join(res_words)
 
 
def speech_to_text():
 
    r = sr.Recognizer()
 
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
 
        st.write("Please say something")
 
        audio = r.listen(source)
 
        st.write("Recognizing Now .... ")
 
 
        # recognize speech using google
 
        try:
            st.write("You have said \n" + r.recognize_google(audio))
            st.write("Audio Recorded Successfully \n ")
 
 
        except Exception as e:
            st.write("Could not hear you please repeat")
            return ""
            # print("Error :  " + str(e))

    return r.recognize_google(audio)

message = speech_to_text()
message = message.lower()
message = lemmatize_sentence(message)

print('message :',message)

words = message.split(' ')
emojis = {
    "smile":"😃",
    "happy":"😄",
    "sad":"😔",
    "angry":"😡",
    "anger":"😡",
    "cry":"😭",
    "laugh":"😂",
    "lol":"😂",
    "fear":"😨",
    "disgust":"🤢",
    "surprise":"😯",
    "freeze":"🥶",
    "froze":"🥶",
    "birthday":"🥳",
    "salute":"🫡",
    "money":"🤑",
    "tasty ":"😋",
    "shy":"🙈",
    "love":"❤️",
    "heart":"❤️", 
    "sleep":"😴"

}

output = ""
for emoji in emojis:
    if emoji in message.lower():
        output += emojis.get(emoji) + " "
print(output)

if output == "":
  #st.write("🙂")
  font_size = "<h1 style='font-size: 148px;'>" + "🙂" + "</h1>"
  st.write(font_size, unsafe_allow_html=True)
else:
  #st.write(output)
  font_size = "<h1 style='font-size: 148px;'>" + output + "</h1>"
  st.write(font_size, unsafe_allow_html=True)