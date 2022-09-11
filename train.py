#!/usr/bin/env python
# coding: utf-8

# In[66]:


import json
from nltk_utils import tokenize,stem
from nltk_utils import bag_of_words


# In[67]:


import numpy as np


# In[68]:


from model import NeuralNet


# In[69]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[70]:


with open('intents.json','r') as f:
    intents= json.load(f)


# In[71]:


print(intents)


# In[72]:


all_words=[]
tags=[]
xy=[]


# In[73]:


#tokenizing
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))


# In[74]:


print(tags)


# In[75]:


print(all_words)


# In[76]:


print(xy)


# In[77]:


ignore_words=['?','.','!',',']


# In[78]:


all_words= [stem(w) for w in all_words if w not in ignore_words]


# In[79]:


print(all_words)


# In[80]:


all_words=sorted(set(all_words))
tags= sorted(set(tags))


# In[81]:


print(tags)


# In[82]:


x_train=[] #bag of words
y_train=[] #tags


# In[83]:


for (pattern_sentence, tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    label= tags.index(tag)
    y_train.append(label)


# In[84]:


x_train= np.array(x_train)
y_train=np.array(y_train)


# In[85]:


#pytorch dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples= len(x_train)
        self.x_data= x_train
        self.y_data= y_train
        
    def __getitem__(self,idx):
        return self.x_data[idx],self.y_data[idx]
        
    def __len__(self):
        return self.n_samples


# In[86]:


dataset= ChatDataset()


# In[111]:


train_loader= DataLoader(dataset=dataset,batch_size=10, shuffle=True, num_workers=2)


# In[112]:


hidden_size=8
output_size= len(tags)
input_size= len(x_train[0])
learning_rate=0.001
no_epochs=1000


# In[113]:


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[118]:


print(device)


# In[114]:


model= NeuralNet(input_size,hidden_size,output_size).to(device)


# In[115]:


#loss and optimizer
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[116]:


print(train_loader)


# In[117]:


for epochs in range(no_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)
        
        #forward
        outputs= model(words)
        loss=criterion(outputs,labels)
        
        #backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1)%100==0:
        print(f'epoch{epoch+1}/{no_epochs}, loss={loss.item():.4f}')
     
print(f'final loss, loss={loss.item():.4f}')
    
        


# In[ ]:




