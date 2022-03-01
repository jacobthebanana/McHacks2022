# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm

# %%
import pandas as pd

# %%
print("Loading recipes.csv")
recipes_df = pd.read_csv('foodcom-recipes-and-reviews/recipes.csv')
print("Loading reviews.csv")
reviews_df = pd.read_csv('foodcom-recipes-and-reviews/reviews.csv')

# %%
recipe_cols = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'Description', 'Images',\
              'RecipeCategory', 'Keywords', 'RecipeIngredientQuantities', 'RecipeIngredientParts', \
              'AggregatedRating', 'Calories', 'RecipeServings', 'RecipeYield', 'RecipeInstructions']
review_cols = ['ReviewId', 'RecipeId', 'Rating', 'Review']

# %%
# limit recipe entries
rp_df = recipes_df[recipe_cols]
rv_df = reviews_df[review_cols]
# extract rows with reviews
rp_df = pd.DataFrame.merge(rp_df, rv_df.RecipeId, on='RecipeId').drop_duplicates('RecipeId')

# %%
recipe_col_subset = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'Description', 'Images',\
              'RecipeCategory', 'Keywords', 'RecipeIngredientQuantities', 'RecipeIngredientParts', \
              'AggregatedRating', 'Calories', 'RecipeInstructions']

rp_df = rp_df.dropna(subset=recipe_col_subset)
rp_df = rp_df[rp_df.Images != 'character(0)']
rp_df.info()

# %%
# reduce rows
# rp_df = rp_df[:1000]
rv_df = pd.DataFrame.merge(rv_df, rp_df.RecipeId, how='inner')

# %%
# This take about 35s with 10000 rows
# RecipeId as a key, list of dict as a value
rc_rv_dict = dict()
for name, df in tqdm(rv_df.groupby('RecipeId')):
    # drop unnecessary column, set index and transpose (to create dict)
    df = df.drop('RecipeId', axis=1).set_index('ReviewId').T
    dict1 = df.to_dict('dict')
    rc_rv_dict[name] = dict1

# %%
# add a new column
rp_df['reviews_in_dict'] = rp_df.RecipeId.apply(lambda x: rc_rv_dict[x])

# %%
import re

def map_str_to_list(string):
    #pattern = re.compile(r'\"(.+)\"')
    pattern = re.compile(r'\"([^"]+)\"')
    return pattern.findall(string)

def map_for_series(series: pd.Series):
    return series.apply(lambda i: map_str_to_list(i))

# %%
modify_col_list = ['Images', 'Keywords', 'RecipeIngredientQuantities', 'RecipeIngredientParts', 'RecipeInstructions']
rp_df = rp_df.apply(lambda x: map_for_series(x) if x.name in modify_col_list else x)

# %%
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)
  
  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label

# %%
def join_instructions(instruction_steps):
    return " ".join(instruction_steps)

# %%

# %%
instructions_encoded = []
instructions = rp_df["RecipeInstructions"].map(join_instructions).dropna().tolist()
for instruction in tqdm(instructions):
    instructions_encoded.append(
        tokenizer.encode(instruction[:512], return_tensors="pt", truncation=True)
    )

# %%
labels = []
for instruction_encoded in tqdm(instructions_encoded):
    output = model.generate(input_ids=instruction_encoded, max_length=2)
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0].split()[1]
    labels.append(label)

# %%
rp_df["emotions"] = labels

# %%
import json
json_output = json.dumps(rp_df.to_dict("records"), indent=2)

with open ("recipe-labelled-full.json", "w") as output_file:
    output_file.write(json_output)

