import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import streamlit as st

lemmatizer = WordNetLemmatizer()
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
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)
 
 
def speech_to_text():
 
    r = sr.Recognizer()
 
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
 
        st.write("Please say something")
 
        audio = r.listen(source)
 
        st.write("Recognizing Now .... ")

        try:
            st.write("You have said \n" + r.recognize_google(audio))
            print("Audio Recorded Successfully \n ")
 
 
        except Exception as e:
            print("Error :  " + str(e))

    return r.recognize_google(audio)
  
  
def speech_to_emoji():

  message = speech_to_text()
  message = message.lower()
  message = lemmatize_sentence(message)

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
      "suprise":"😯",
      "freeze":"🥶",
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
  for word in words:
      if word in emojis:
          output += emojis.get(word,word) + " "

  if output == "":
    return "🙂"
  else:
    return output