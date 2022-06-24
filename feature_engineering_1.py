import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#import data
reviews = pd.read_csv("reviews.csv")
 
#print column names
reviews.columns
 
#print .info
reviews.info()

#look at the counts of recommended
print(reviews["recommended"].value_counts())
 
#create binary dictionary
binary_dictionary ={True:1, False:0} 

 
#transform column
reviews["recommended"] = reviews["recommended"] .map(binary_dictionary)
print(reviews["recommended"].value_counts())
 
#print your transformed column
print(reviews["recommended"])

#look at the counts of rating
print(reviews["rating"].value_counts())
 
#create dictionary
rating_dict = {"Loved it":5,"‘Liked it":4,"Was okay":3,"Not great":2,"Hated it":1}
 
#transform rating column
reviews["rating"] = reviews["rating"] .map(rating_dict)
print(reviews["rating"].value_counts())

#print your transformed column values
print(reviews["rating"])

#get the number of categories in a feature
print(reviews["department_name"].value_counts())
 
#perform get_dummies
one_hot = pd.get_dummies(reviews["department_name"])
print(one_hot)

#join the new columns back onto the original
reviews =reviews.join(one_hot)

#print column names
print(reviews.columns)

#transform review_date to date-time data
reviews["review_date"] = pd.to_datetime(reviews["review_date"])

#print review_date data type 
print(reviews["review_date"].dtypes)

#get numerical columns
reviews = reviews[['clothing_id', 'age', 'recommended', 'rating', 'Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend']].copy()
 
#reset index
reviews = reviews.set_index("clothing_id")

#instantiate standard scaler
standard_scalar =StandardScaler()
 
#fit transform data
standard_scalar.fit_transform(reviews)

#data used from kaggle
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

