from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re

# gets all of the career links

request = Request("https://collegegrad.com/careers/all")
html_page = urlopen(request)
soup = BeautifulSoup(html_page, "lxml")

links = []
for link in soup.findAll('a'):
    if link.get('href').find("/")==-1: 
      continue 
    else: 
      url = "https://collegegrad.com" + link.get('href')
    links.append(url)

career_links = []

for x in links:
  if x.find("careers")==-1 or len(x.split("https://collegegrad.com/careers/"))==1:
    continue
  else: 
    career_links.append(x) 
    
!pip install transformers
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# loads the Bert Model and Tokenizer 

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

!pip install pyspellchecker
from spellchecker import SpellChecker
import string

# gets user input for career

user_input = input("Enter your careeer:")

# cleans user input for career 

cleaned_user_input = ''
for x in user_input:
  if x not in string.punctuation: 
    cleaned_user_input = cleaned_user_input + x 

career_choice_words = cleaned_user_input.split(" ")

spell_checker = SpellChecker()
misspelled_words = spell_checker.unknown(career_choice_words)

if len(misspelled_words)>0: 
  for x in misspelled_words:
    career_choice_words.remove(x) 


for word in misspelled_words:
    career_choice_words.append(spell_checker.correction(word))

# algorithmically determines the most likely career link based on their career input 

ratio = 0.0
url = ""

for x in career_links:
  career = x.split("https://collegegrad.com/careers/")[1]
  career_list = career.split("-")
  total_count = len(career_list)
  count = 0
  for x in career_choice_words:
    for y in career_list: 
      if x in y: 
        count = count + 1  
  stats = float(count/total_count)
  if stats>ratio:
    ratio = stats
    url = "https://collegegrad.com/careers/" + career

# gets all the text from that link and separates it into four categories 

!pip install inscriptis
import urllib.request 
from inscriptis import get_text 

html_page = urllib.request.urlopen(url).read().decode('utf-8') 
 
text = get_text(html_page) 

indices = [] 

key = "[ About this section ]"
start = 0

while text.find(key, start)!=-1:
  index = text.find(key, start)
  indices.append(index)
  start = index + len(key)

indices = indices[:-2] 

# labels each category and stores the corresponding text for that category from the chosen article 

dictionnary = {"What They Do":0, "Work Environment":1, "Job Qualifications":2, "Salary":3}  

category_text = []

for x in range(0, len(indices)-1):
  category_text.append(text[indices[x]+len(key):indices[x+1]]) 
  
# gets the user input for the category 

user_category = input("Select a category (What They Do, Work Environment, Job Qualifications, Salary)") 

ref_text = category_text[dictionnary[user_category]]


# uses machine learning with BERT to create a question-answer system that selects the most likely answer from the question you just asked. 

import string

question = input("Enter in your career question:") 

reference =  ref_text
            
encoding = tokenizer.encode_plus(text=question, text_pair=reference)

inputs = encoding['input_ids']
sentence_embedding = encoding['token_type_ids']
tokens = tokenizer.convert_ids_to_tokens(inputs)

results = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

answer = ''

for x in range(torch.argmax(results[0]), torch.argmax(results[1])+1):
  if tokens[x] in string.punctuation:
    answer = answer + tokens[x]
  else: 
    answer = answer + " " + tokens[x]

print(answer)
