# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:35:15 2023

@author: admin
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import tensorflow as tf
tf.test.gpu_device_name
tf.config.experimental.list_physical_devices('GPU')
tf.__version__
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")

############ Standard Library for standard scaling #######################

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_pipelines = Pipeline([('min_scaler', MinMaxScaler())])

##########################################################################
###### creating benign only text file
seqentries  = 20 # number of entries to keep in one window
file = pd.read_csv(r"C:\Users\admin\Downloads\new_dataset.csv (1)\new_dataset.csv")

api_call_seq_enum = []
for i in range(0,len(file)):
    if(file.iloc[i]['malware']==0): # selecting those sequences which are labaled malicious
        tempseq = list(file.iloc[i])
        tempseq.pop()
        tempseq = tempseq[2:] # remove label and other entries like md5
        tempseq_nan_removed = [x for x in tempseq if str(x) != 'nan']
        if(len(tempseq_nan_removed)>=seqentries):
            tempseq = [int(i) for i in tempseq_nan_removed]
            tempseq = [str(x) for x in tempseq]
            api_call_seq_enum.append(tempseq[0:seqentries])
### Writing refined malicious sequence to the api_seq_enum text file        
files = open('api_seq_enum.txt', 'w')
for j in range(0,len(api_call_seq_enum)):
     files.write(' '.join(api_call_seq_enum[j]))
     files.write('\n')
files.close()
### Reading the files
f = open(r'C:\Users\admin\Downloads\SeqGAN-master\SeqGAN-master\save\real_data.txt','r')
contents = f.readlines()
f1 = open(r'C:\Users\admin\api_seq_enum.txt','r')
contents1 = f1.readlines()
### Clean similar generated samples from seqGAN
f_adversarial = open(r'G:\Chapter_1_work\SeqGAN-master\save\generator_sample.txt','r')
contents_to_add = f_adversarial.readlines()
def delete_similar_strings(string_list):
    unique_strings = []
    for string in string_list:
        if not any(similar_string(string, unique_str) for unique_str in unique_strings):
            unique_strings.append(string)
    return unique_strings
def similar_string(string1, string2):
    # Implement your similarity comparison logic here
    # This can be a simple string comparison or more advanced techniques like Levenshtein distance or fuzzy matching
    # Return True if the strings are similar, otherwise False
    return string1 == string2
unique_strings = delete_similar_strings(contents_to_add)
### adversarial sample create by combining generated strings and real malicious sequences
### Below code for perturbation at last, use random() for perturbing at other place
c = open('generated_adversarial_trial_6.txt','w')
for x in range(0,len(api_call_seq_enum)):
    str1 = api_call_seq_enum[x]
    for z in range(0,len(unique_strings)): # unique_strings is generated sequence
        str2 = contents_to_add[z]
        str3  = str1 + " " + str2 # use random() for adding string at different index
        c.write(str3)
c.close()  
# adversarial dataset pre creation steps
gad = open('generated_adversarial_trial_6.txt','r')
contentsgad = gad.readlines()
gad.close()  
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(3, 3))
X2 = vectorizer2.fit_transform(contentsgad)
feature_names  = vectorizer2.get_feature_names()
dataset_api_3_gram_adversarial_feature  = X2.toarray()
dataset_api_3_gram_adv_feature = pd.DataFrame(dataset_api_3_gram_adversarial_feature)
#### removal of features before creating datase creation
original_dataset = pd.read_csv(r"C:\Users\admin\selected_columns.csv")
originalcolumns = list(original_dataset.columns)
preadversary = feature_names
indextodel=[]
for s in range(0,len(preadversary)):
    if(preadversary[s] not in originalcolumns):
        indextodel.append(s) 
#### creation of adversarial attack dataset
adv_dataset = dataset_api_3_gram_adv_feature.drop(indextodel,axis = 1)
adv_dataset.to_csv('adv_dataset.csv')
###### trained model loading
adv_dataset  = pd.read_csv('G:\\Chapter_1_work\\adv_dataset.csv')
model = tf.saved_model.load('C:\\Users\\admin\\saved_model.pb')
col_adv = adv_dataset.columns
adv_dataset = adv_dataset.drop("Unnamed: 0", axis=1) 
adv_dataset_numpy_version = adv_dataset.to_numpy
avast_adv_test = adv_dataset.values.reshape(254722,105,105)
avast_adv_test=avast_adv_test[...,np.newaxis]
score = model.predict(avast_adv_test)
score_lab=[]
for i in range(0,len(score)):
    score_lab.append(np.argmax(score[i]))  
file = open('adv_attack_result.txt','w')
for item in score_lab:
	file.write(f"{item}\n")
file.close()
### testing the results of original malicious samples
selected_columns = pd.read_csv("C:\\Users\\admin\\selected_columns.csv")
selected_columns.drop(selected_columns[selected_columns.label == 0].index, inplace=True)
all_malw_samples= selected_columns.drop("label", axis=1) 
housing_prepared = all_malw_samples.to_numpy()
all_malw_samples_accum = housing_prepared.reshape((2626,105,105))
all_malw_samples_accum = all_malw_samples_accum[...,np.newaxis]
score = model.predict(all_malw_samples_accum)
score_lab=[]
for i in range(0,len(score)):
    score_lab.append(np.argmax(score[i]))  
final_score_accm_malware = []
for i in range(0,len(score_lab)):
    for j in range(0,97):
        final_score_accm_malware.append(score_lab[i])
file = open('all_mal_pred_result.txt','w')
for item in final_score_accm_malware:
	file.write(f"{item}\n")
file.close()  