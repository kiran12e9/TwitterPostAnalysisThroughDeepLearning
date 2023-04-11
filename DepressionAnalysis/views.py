from datetime import timezone
from imp import load_module
import json
import tokenize
from django.shortcuts import redirect, render,HttpResponse
import numpy
from sklearn import model_selection
from .models import Biodata
from keras.preprocessing.text import Tokenizer 
import tensorflow as tf
from django.contrib.auth import authenticate, login , logout
import random
import openai 
import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re


import hashlib
from math import exp
from numpy import sign

from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer
import keras
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import plot_model



import pickle
import tensorflow as tf
import pickle
import joblib
from django.contrib.auth.decorators import login_required
import joblib

import tweepy

status="Please scroll down to view the analysis..."
tweet=""
model22=""
cleaned_tweet= []
clean_tweets=""

from keras.models import load_model
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

quotes=[
"Please remember that you’re capable, brave and loved – even when it feels like you’re not.",
"Do not give the past the power to define your future.",
"Perhaps you’ve been assigned this mountain to show others that it can be moved.",
"Just a reminder: it is NOT selfish to put your recovery first. Rather, it’s necessary in order to make sure that everything else doesn’t come last.",
"Small, baby steps each day add up to huge, giant leaps over time. So, please keep going. Do NOT give up.",
"You are not worthless, you are not a failure, and you are not a loser. That voice saying you are is just your depression trying to trick you.",
"On those really difficult days when it seems impossible to go on and you feel like giving up, just remind yourself that you’ve been there before and you’ve survived every time, so you can survive this time, too.",
"Even the darkest hour only has 60 minutes.",
"Perhaps the butterfly is proof that you can go through a great deal of darkness yet become something beautiful again.",
"Always try to end the day with a positive thought. No matter how hard things are, tomorrow is a fresh opportunity to make everything better.",
"The World Health Organisation estimates that 350 million people suffer from depression worldwide. We know it may not seem like it, but you are NOT alone.",
"Don’t hate yourself for everything you aren’t. Instead, love yourself for everything you are.",
"Running away from your problems is a race you’ll never win. Instead, reach out for help and try to confront them.",
"Don’t let your struggle become your identity. After all, you are so much more than just your illness.",
"On your good days, write down your reasons to keep on fighting. Then on your bad days, read over your list to give you strength.",
"Be proud of who you are, instead of ashamed of how someone else sees you.",
"Crying doesn’t mean that you’re weak. Since birth, it’s always been a sign that you’re alive.",
"Don’t dwell on those who hold you down. Instead, cherish those who helped you up.",
"Never, ever, ever, ever, ever give up on yourself. As long as you keep on fighting, then you can beat your depression.",
"Right now, stop whatever you’re doing and think of all the things in life that you are grateful for. This is a really easy way to lift your mood!",
"Even if you can’t see any reason to keep on going, then it doesn’t mean that there aren’t any. It just means that in that moment, your depression is telling you even more lies than usual.",
"Even the worst depressive episodes won’t last forever.",
"Having depression does not mean you are weak, a failure, or worth less than anybody else. Please, don’t discriminate against yourself.",
"If you need a confidence booster, then remind yourself of all the difficult things you’ve endured and overcome.",
"Don’t ruin a good day by thinking about the possibility of a bad day in the future. Just enjoy the present moment :)",
"As Confucius said, our greatest glory is not in never falling, but in rising every time we do.",
"When your depression says, “Give up”, hope whispers, “Try one more time”.",
"If you’re worried about telling a friend that you struggle with depression, then remind yourself that you are one of 350 million people battling this illness. So, odds are that even if you don’t know it, one or more of your friends are fighting depression too but are similarly scared to reach out to you.",
"You are brave, courageous and strong for continuing to fight an illness that nobody else can see."]

def form(request):
    
    if(request.method=='POST'):
        
        tweet=request.POST.get('tweet')
        # Expand Contraction
        tweetoriginal=tweet
        
        c_re = re.compile('(%s)' % '|'.join(cList.keys()))

        def expandContractions(text, c_re=c_re):
            def replace(match):
                return cList[match.group(0)]
            return c_re.sub(replace, text)
       

        def clean_tweets(tweets):
                cleaned_tweets = []
                for tweet in tweets:
                        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
                        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            # remove hashtag, @mention, emoji and image URLs
                                tweet = ' '.join(
                                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
                                tweet = ftfy.fix_text(tweet)

            # expand contraction
                                tweet = expandContractions(tweet)

            # remove punctuation
                                tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
                                stop_words = set(stopwords.words('english'))
                                word_tokens = nltk.word_tokenize(tweet)
                                filtered_sentence = [w for w in word_tokens if not w in stop_words]
                                tweet = ' '.join(filtered_sentence)

            # stemming words
                                tweet = PorterStemmer().stem(tweet)

                                cleaned_tweets.append(tweet)

                return cleaned_tweet
        tokenizer = pickle.load(open(r'C:\\Users\\rkira\\tokenizer.pkl', 'rb'))
        
        
        sequences = tokenizer.texts_to_sequences([tweet])

        max_sequence_len = 140
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
        modeldummy=load_model(r'C:\\Users\\rkira\\weights600.h5')
        
     
       
        resfinal = modeldummy.predict(padded_sequences)
        prediction_class =resfinal # make prediction for each tweet
        datas=['res']
        if resfinal[0][0]>=0.7 :
                    resfinal="severe derpression"
                    return render(request, 'result.html', {'res':resfinal,'te':"1",'quotes':json.dumps(quotes),'status':status,'dep':datas})
        
        elif resfinal[0][0]>=0.45 and resfinal[0][0]<=0.69 :
                    resfinal="mild depression"
        else:
                    resfinal="no depression symptoms"            
                    return render(request, 'result.html', {'res':resfinal,'te':"1",'quotes':json.dumps(quotes),'status':status,'nodep':datas})
        
        


    else: 
          if request.user.is_authenticated:
                
            return render(request,'form.html',{'quotes':json.dumps(quotes)})
          return redirect(userlogin)    
    
         



def tweet(request):
    if(request.method=='POST'):
     
        usernaam=request.POST.get('usernaam')
        # Expand Contraction
        try:
            api_key="j46JUY3EXSBaHvV08d79SpO77"
            api_key_secret="kGA7SjRizBiKbLLYDAu0qr5ItIXJqyaggjJ1IK4zltdmqSE5Qt"
            access_token="1584054523038863360-u8DtupXd4GanJTsLpigc2CAVHfbrR7"
            access_token_secret="DUpHPTM0tKWHjzAml422CvlHLfIhIs6Ov28lDmXbCAaWq"
            auth=tweepy.OAuthHandler(api_key,api_key_secret)
            auth.set_access_token(access_token,access_token_secret)
            api=tweepy.API(auth)
            print("failed here1")
            user=usernaam
            print("failed here2")
            limit="10"
            print("failed here3")
            tweets2=api.user_timeline(screen_name=user,count=limit,tweet_mode='extended')
            print("failed here4")
            data=""
            print("failed here5")
            for tweet1 in tweets2:
                data=data+str(tweet1)
                tweet=data

            print("failed here6")
            print(tweet)
        except:
            return render(request,'tweet.html',{'er':'Error while Accessing Twitter API, Request Quota Exceeded.'})    
        c_re = re.compile('(%s)' % '|'.join(cList.keys()))

        def expandContractions(text, c_re=c_re):
            def replace(match):
                return cList[match.group(0)]
            return c_re.sub(replace, text)
       

        def clean_tweets(tweets):
                
                for tweet in tweets:
                        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
                        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            # remove hashtag, @mention, emoji and image URLs
                                tweet = ' '.join(
                                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
                                tweet = ftfy.fix_text(tweet)

            # expand contraction
                                tweet = expandContractions(tweet)

            # remove punctuation
                                tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
                                stop_words = set(stopwords.words('english'))
                                word_tokens = nltk.word_tokenize(tweet)
                                filtered_sentence = [w for w in word_tokens if not w in stop_words]
                                tweet = ' '.join(filtered_sentence)

            # stemming words
                                tweet = PorterStemmer().stem(tweet)

                                cleaned_tweet.append(tweet)

                return cleaned_tweet
        tokenizer = pickle.load(open(r'C:\\Users\\rkira\\tokenizer.pkl', 'rb'))
        
        
        sequences = tokenizer.texts_to_sequences([tweet])

        max_sequence_len = 140
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
        modeldummy=load_model(r'C:\\Users\\rkira\\weights600.h5')
     
       
        resfinal = modeldummy.predict(padded_sequences)
        datas=['res']
        prediction_class =resfinal # make prediction for each tweet    
        if resfinal[0][0]>=0.7 :
                    resfinal="severe derpression"
                    return render(request, 'result.html', {'res':resfinal,'tw':"1",'quotes':json.dumps(quotes),'usernaam':usernaam,'status':status,'dep':datas})
        
        elif resfinal[0][0]>=0.45 and resfinal[0][0]<=0.69 :
                    resfinal="mild depression"
        else:
                    resfinal="no depression symptoms"            
                    return render(request, 'result.html', {'res':resfinal,'tw':"1",'quotes':json.dumps(quotes),'usernaam':usernaam,'status':status,'nodep':datas})
        
        
     
 
    else: 
          if request.user.is_authenticated:
                
            return render(request,'tweet.html',{'quotes':json.dumps(quotes)})
          return redirect(userlogin)

def signup(request):
    if request.method=='POST':
        emailid=request.POST.get('emailid1')
        password=request.POST.get('password1')
        cpassword=request.POST.get('cpassword1')
        email_bytes = emailid.encode('utf-8')
        sha256 = hashlib.sha256()
        sha256.update(email_bytes)
        username = sha256.hexdigest()
        b=Biodata(email=emailid,username=username,password=password)
        try:
            if Biodata.objects.filter(username=username).exists():
                error="email id already in use, please login or click on forgot password to reset your password"
                return render(request,'signup.html',{'error':error})
            b.save()
            return redirect('userlogin')
        except:
            print("details not saved")
            return render(request,'signup.html')
    else:
        if request.user.is_authenticated:
              return redirect('tweet')
        else:
              print(request.user.is_authenticated)
              return render(request,'signup.html')
    
def userlogin(request):
    if request.method=='POST':
        emailid=request.POST.get('email')
        password=request.POST.get('password')
        try:
            user = Biodata.objects.get(email=emailid)
        except:
            return render(request,'login.html')
        stored_password = user.password
        if stored_password == password:
            login(request,user)
            return redirect('tweet')
        else:
            error="Invalid login Credentials"
            return render(request,'login.html',{'error':error})      
    else:
        return render(request,'login.html')
    


def userlogout(request):
    logout(request)
    return render(request,'login.html')    
    


def home(request):
    if request.user.is_authenticated:
          return redirect('tweet')
    else:
          return render(request,'home.html')

