#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:15:16 2018

@author: yuyingjie
"""
from __future__ import unicode_literals, print_function
import sys
import glob
import errno
import PyPDF2
import os
from nltk import sent_tokenize
import spacy
from spacy.tokens import Span
import re

path1 = '/Users/yuyingjie/elliemae/US Bank JSON files/*'   
folders = glob.glob(path1)
pdfs = {}   
for name in folders: 
        pdf_path = os.path.join(name,"*.pdf")
        file = glob.glob(pdf_path)[0]
        pdf_file = open(file,'rb')
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        file_id = name.split('/')[-1]
        pdfs[file_id] = []
        for i in range(read_pdf.getNumPages()):
            page = read_pdf.getPage(i)
            page_content = page.extractText()
            pdfs[file_id].append(page_content)

            
print(len(pdfs['6544']))            
            
for k,v in pdfs.items():
    for i in range(len(pdfs[k])):
        page_content = pdfs[k][i]
        page =  sent_tokenize(page_content)
        page = [item.strip().replace('\n',"") for item in page]
        pdfs[k][i] = page
        
#%%
'''
Here we check if fico is identified as name entity in one doc
'''
file = pdfs['6544']
fico_indices={}
for i in range(0,len(file)):
    file[i] = [str.lower(x) for x in file[i]]
    fico_indices[i] =  [index for index, s in enumerate(file[i]) if 'fico' in s]
print(fico_indices)

fico_doc = []
for k, v in fico_indices.items():
    if len(v)!=0:
        for i in v:
            fico_doc.append(file[k][i])
fico_doc

nlp = spacy.load('en_core_web_sm')
doc = nlp(fico_doc[1])
doc
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)
# the model didn't recognise "fico" as an entity 
#%% creating training data
'''
for trainig data, we extract all sentences that including 'fico' as well as the corresponding index in sentences
The format of TRAIN_DATA  is a list: [(sentence, {'entities':[(index.start,index.end, LABEL)]})]
'''


import random
from pathlib import Path

TRAIN_DATA =[]
# new entity label
LABEL = 'CREDIT'

for id in pdfs.keys():
    file = pdfs[id]
    fico_indices={}
    for i in range(0,len(file)):
        file[i] = [str.lower(x) for x in file[i]]
        fico_indices[i] =  [index for index, s in enumerate(file[i]) if 'fico' in s]
    fico_doc = []
    for k, v in fico_indices.items():
        if len(v)!=0:
            for i in v:
                fico_doc.append(file[k][i])
    for doc in fico_doc:
        index = re.search(r'\b(fico)\b', doc)
        TRAIN_DATA.append((doc, {'entities':[(index.start(),index.end(), LABEL)]}))
TRAIN_DATA[0]

#%% training model
# code copied from Spacy's github

def add_new_entity(model=None, new_model_name=LABEL, output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()



    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'the minimum required fico score is...'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


add_new_entity(model='en_core_web_sm', new_model_name=LABEL, output_dir=None, n_iter=20)


#%% test
CREDIT = doc.vocab.strings[u'CREDIT']  # get hash value of entity label
a = re.search(r'\b(fico)\b', fico_doc[1])
print(a.end())
fico_ent = Span(doc, a.start(),a.end(), label=CREDIT) # create a Span for the new entity
doc.ents = list(doc.ents) + [fico_ent]

nlp = spacy.load('en_core_web_sm')




