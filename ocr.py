#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Siddartha Rao, Vishal Singh, Jai Kumar )
# (based on skeleton code by D. Crandall, Oct 2018)
#

##The variables used in training are-

#Initial probability(ps1)

# This is probability of each character occuring at first position. It is the frequency

# that each character at first position by the total count. The values are stored in

#'log' values since all calculations are performed in log to avoid underflow

 

#State(ps)

#This is the probability of a character occurring irrespective of the position except

#for the characters occuring at the first position. It is frequency of each character divided

#by total number of characters. The values are stored in log

#

#Transition(pss)

#This is probability of two characters occurring in sequence. It is frequency of all

#character pairs in sequence divided by total number of characters pairs. The values are

#stored in log. pss variable is a dictionary which contain pair of characters as pairs

#e.g.P[('character1','character2')].

#Please note that ‘pss’ stores the probabilities that two characters occur in a sequence.

#This is P(S and Si+1). To find P(Si+1|Si) we have to subtract P(Si) from P(S and Si+1).

#

#Emission()

#This probability is calculated by comparing the test image with the train image.

#p is the number of matched pixels

#m is the noisy pixel, we assume this value to be around (0.1 to 0.2)

#prob =  (1-m)^p*(m)^(350-p)

#The values are stored in log

#NAIVE BAYES(SIMPLE)

#This is implemented by maximising the posterior probability(check posterior

#probability section) for each train character given a test character. In case the test data has an

#unseen character. A small probability is substituted 

#VITERBI(HMM)

#The algorithm is implemented by initializing two  dictionary of dictionaries

#'viterbi' and 'backtrack' the keys of outer dictionary are the unique characters

#and the keys of the inner dict are the characters of the test image. The viterbi dict

#stores the max score obtained by implementing the viterbi algorithm at a

#particular state and the backtrack dict stores the value of character from which

#the maximum score is obtained(argmax). To generate a sequence the charatcer is obtained

#from the viterbi dict whose probability is maximum for the last word, this is

#the last character of the given sentence. corresponding values is obtained from

#backtrack dict, this is the chaaracte of the the second last word. The charcaters are traced

#back from the backtrack dict. This order is then reversed to obtain the sequence

#0f characters for the given sentence


from PIL import Image, ImageDraw, ImageFont
import sys
from collections import Counter
import operator
from copy import deepcopy
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

# The function resds the test immage and returns the image in a pixel format
#Written by D. Crandall
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

#this function reads the train-image and returns the 72 caharacters in a pixel format 25*14
#Written by D. Crandall
def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# This function is used to read the train data set, used the same code as in part one
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    return exemplars

#In this function we calculate the different probabilites, like 
#ps1 --> probability of the  character in the first position
#ps  --> probability of the character in the whole dataframe 
#pss --> which is the transition probabilites     
def train(data,train_letters):
    
    #P(S1)
    letter1=[]
    for i in range(len(data)):
        if(data[i][0][0][0] in train_letters):
            letter1.append(data[i][0][0][0])
                
    letter_uniq = list(set(letter1))
    missing = list(set(train_letters) - set(letter_uniq))
    for i in missing:
        letter_uniq.append(i)
        letter1.append(i)
    
    ps1=dict(Counter(letter1))
    ps1 = {k:math.log( v / total) for total in (sum(ps1.values(), 0.0),) for k, v in ps1.items()}
    
    #P(S)
    all_Letters=[]
    for i in range(len(data)):
        first=0 
        for word in (data[i][0][:]) :
            for s in word:
                if(s in train_letters):
                    if(first==0):
                        first=1
                    else:
                        all_Letters.append(s)
                        
            all_Letters.append(" ")       
        all_Letters.pop()    
    letterAll_uniq = list(set(all_Letters))
    missing = list(set(train_letters) - set(letterAll_uniq))
    for i in missing:
        letterAll_uniq.append(i)
        all_Letters.append(i)
    ps=dict(Counter(all_Letters))    
    ps = {k: math.log(v / total) for total in (sum(ps.values(), 0.0),) for k, v in ps.items()}
 
    #P(Si+1|Si)
    pss={}
    transition=[] 
    for i in range(len(data)):
        first=0 
        for word in (data[i][0][:]) :
            for s in word:
                if(s in train_letters):
                    if(first==0):
                        first=1
                        t1= s
                        t2=s
                    else:
                        t2=s
                        t=(t1,t2)
                        transition.append(t)
                        t1=t2
            t2=" " 
            t=(t1,t2)
            t1=t2
            transition.append(t)  
        transition.pop()    
    allComb=[]
    for p in train_letters:
       for q in train_letters:
           t=(p,q)
           allComb.append(t)
    
    comb_uniq = list(set(transition))
    missing = list(set(allComb) - set(comb_uniq))
    for i in missing:
        comb_uniq.append(i)
        transition.append(i)
           
    pss=dict(Counter(transition))
    pss = {k:math.log( v / total) for total in (sum(pss.values(), 0.0),) for k, v in pss.items()}
    #P(Si+1|Si)
    return(ps1,ps,pss,letterAll_uniq)
        
# this is the first basic emmision , here we try to check for all the dark pixels that are matching  
#This function returns the number of dark pixels matched   
def emmision1(test):
    c=0
    dictLetter= {}
    
    for i in train_letters:
        for j in range(0,len(train_letters[i])):
            for k in range(0,len(train_letters[i][j])):
                if(train_letters[i][j][k]==test[j][k]=="*"):
                    c+=1
        dictLetter[i]=c
        c=0        
    maximum = max(dictLetter, key=dictLetter.get)  
    if dictLetter[maximum]!=0:
        print()
    else :   
        key1= ' '
        print(key1,dictLetter[' '])
    return(dictLetter)    

#This is the emmision that is used in bayes,
# Here we checck for all the pixels matched( both dark and white)
# the prob= (1-m)^p*(m)^(350-p)
# where m = the error and p= matched pixels    
def emmision2(test):
    c=0
    let={}
    dictLetter= {}
    
    for i in train_letters:
        for j in range(0,len(train_letters[i])):
            for k in range(0,len(train_letters[i][j])):
                if(train_letters[i][j][k]==test[j][k]):
                    c+=1
        let[i]=c
        dictLetter[i]=math.log(((1-.15)**c)*(.01)**(350-c))
        c=0   
    return(dictLetter)    

#In this emmsion we identifed 4 different types of checks
#1. dark pixel of train = dark pixel of test
#1. white pixel of train = white pixel of test
#1. dark pixel of train = white pixel of test
#1. white pixel of train = dark pixel of test
    
def emmision3(test):
    c=0
    d=0
    noise=0
    noiseless=0
    dictLetter= {}
    for i in train_letters:
        for j in range(0,len(train_letters[i])):
            for k in range(0,len(train_letters[i][j])):
                       
                if(train_letters[i][j][k] == "*" and test[j][k]=="*"):
                    c+=1
                elif(train_letters[i][j][k] == " " and test[j][k]==" "):
                    d+=1  
                elif(train_letters[i][j][k] == " " and test[j][k]=="*"):
                    noise+=1
                elif(train_letters[i][j][k] == "*" and test[j][k]==" "):
                    noiseless+=1
        x=350-(c+d)
       # if x<247:x=247
       # if((i== " " and d<335) or (i== "'" and d<319) or (i== "," and d<308) or (i== "." and d<316) or (i== "-" and d<307) or (i== '"' and d<307 ) or (i== '1' and d<296 ) or (i== '!' and d<296 )):
       # if((i== " " and d<320) or ((i== "'"  or i== ","  or i== "."  or i== "-"  or i== '"'  or i== '1' or i== '!' )and d+noiseless+noise!=350)):
        if((i== " " and d<340) or (i== "'" and d+noise+noiseless<=339) or (i== "," and d+noise+noiseless<=328) or (i== "." and d+noise+noiseless<=336) or (i== "-" and d+noise+noiseless<=327) or (i== '"' and d+noise+noiseless<=327 ) or (i== '1' and d+noise+noiseless<=316 ) or (i== '!' and d+noise+noiseless<=316 )):
            dictLetter[i]=-10000000
        else:    
           dictLetter[i]=(math.log(((1-.15)**(c+d))*(.01)**(x)))/10
        c=0   
        d=0
        noise=0
        noiseless=0
    return(dictLetter)    
    
#This is a combination of emmision1 and emmision2
#Here we check for the dark pixels matched and appy the formula used in emmison2   
# the prob= (1-m)^p*(m)^(350-p)
# where m = the error and p= matched pixels    
    
def emmision4(test):
    c=0
    d=0
    dictLetter= {}
    
    for i in train_letters:
        for j in range(0,len(train_letters[i])):
            for k in range(0,len(train_letters[i][j])):
                if(train_letters[i][j][k]==test[j][k]=="*"):
                    c+=1
                elif(train_letters[i][j][k] == " " and test[j][k]==" "):
                    d+=1  
                   
        x=350-c-d
        if((i== " " and (d<339)) or (( i== "'" or i== "," or i== "." or i== "-" or i== '"' or i== '1' or i== '!' )and (d<328)) or(( i== '(' or i== ')') and d<315) ):
             dictLetter[i]=-10000000
        else: 
            dictLetter[i]=math.log(((1-.15)**(c+d))*(.01)**(x))/5
        c=0  
        d=0
        
    return(dictLetter)    
    
def emmisionFunc(i,p):
    if p==2:
     dictLetter=   emmision2(i)
    else:
     dictLetter=   emmision4(i)
    return(dictLetter) 
#This is the function for simple Bayes 
# Here likelihood is the log of matched pixels
# Prior is (ps), which is the probability of the character in the train dataset
def bayes(ps):
    line=[]
    for i in  test_letters:
        likelihood= emmision2(i)
        prob ={k: ps[k]+likelihood[k] for k in ps}
        line.append(max(prob, key=prob.get))
    res = "".join(line)
    print("Simple:", res) 
    
#The viterbi algorithm is implemented in this function
#We make a check to find if the image is noisy or if its charatcers are light.
#check if the first characters has less than 75*,if yes then we call emmision2 else we call emmision4 function
    
    
def hmm_viterbi(ps1,ps,pss,letterAll_uniq):
   
    p=2
    count=0
    viterbi={}
    viterbi = viterbi.fromkeys(letterAll_uniq)
    backtrack={}
    backtrack = backtrack.fromkeys(letterAll_uniq)
    length=len(test_letters)
    listLen=[]
    for j in range(0,len(test_letters[0])):
        for k in range(0,len(test_letters[0][j])):                       
            if(test_letters[0][j][k]=="*"):
                count+=1
    if(count<75):
      p=4         
    
    for i in range (1,length+1):
        listLen.append(i)
    lenDict=dict.fromkeys(listLen)
    first=0
    x=0
    c=0
    for charecters in letterAll_uniq:
        viterbi[charecters] = deepcopy(lenDict)
        backtrack[charecters] = deepcopy(lenDict)
        
    for i in  test_letters:
        x+=1
        for charecters in letterAll_uniq:
            char_dict={}
            emmision= emmisionFunc(i,p)
            char_dict = char_dict.fromkeys(lenDict)
            if first==0:
                viterbi[charecters][x]=emmision[charecters]+ps1[charecters]
                backtrack[charecters][x]=0
            else:
                c=c+1
                max_prob=-10000000
                for characters1 in letterAll_uniq:
                    prob=viterbi[characters1][x-1]+pss[(characters1,charecters)]
                    max_prob=max(max_prob,prob)
                    if max_prob==prob:
                        viterbi[charecters][x]=emmision[charecters]+max_prob
                        backtrack[charecters][x]=characters1    
        first=1
        c=0
    max_prob = -10000000
    finalWord=[]
    for characters in letterAll_uniq: 
         prob = viterbi[characters][length]
         max_prob=max(max_prob,prob)
         if max_prob==prob:
             maxChar= characters
    finalWord.append(maxChar)
    for listLen1 in reversed(listLen[1:]):
        finalWord.append(backtrack[maxChar][listLen1])
        maxChar=backtrack[maxChar][listLen1]
    finalWord = finalWord[::-1] 
    res = "".join(finalWord)
    print("Viterbi:",res)
    print("Final answer:")
    print(res) 
   
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

trainData= read_data(train_txt_fname)
ps1,ps,pss,letterAll_uniq=train(trainData,train_letters)



bayes(ps)

hmm_viterbi(ps1,ps,pss,letterAll_uniq)
