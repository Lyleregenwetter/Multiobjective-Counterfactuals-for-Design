import os
import pandas as pd
import pymoo
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Binary, Choice
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from autogluon.tabular import TabularDataset, TabularPredictor
import torch
import importlib
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import GA_Clip_utils
import datetime
import dill
import warnings
import glob
import textwrap
import imageio
import sys
import decode_mcd_private.calculate_dtai
import load_data
import decode_mcd.counterfactuals_generator
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

biked = pd.read_csv("../CLIP/clip_sBIKED_processed.csv", index_col=0)
dataset = "clip_s"
embeddings = pd.read_csv("../CLIP/img_embeddings.csv", index_col=0)

print(biked)
print(embeddings)

device = "cuda"
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)  # .to(device)

prompt1 = "a futuristic black road racing bicycle"
# prompt2 = "an orange cyberpunk-style lowrider bike"
inputs1 = tokenizer(prompt1, return_tensors="pt")  # .to(device)
target_embedding1 = model.get_text_features(**inputs1)  # Our target text embedding
# inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)
# target_embedding2 = model.get_text_features(**inputs2) #Our target text embedding

targetfile = "mtb.png"
img = Image.open(targetfile).convert("RGB")
width, height = img.size
img = img.resize((width // 2, height // 2))
result = Image.new(img.mode, (1300, 1300), (255, 255, 255))
result.paste(img, img.getbbox())
image = np.asarray(result)
img_processed = processor(text=None, images=image, return_tensors='pt')['pixel_values']  # .to(device)
target_embedding2 = model.get_image_features(img_processed)

a = torch.tensor(embeddings.values)
b1 = target_embedding1.cpu()
b2 = target_embedding2.cpu()
cos = torch.nn.CosineSimilarity()
res1 = cos(a, b1)
res1 = res1.detach().numpy()
res2 = cos(a, b2)
res2 = res2.detach().numpy()
targetval_1 = np.quantile(res1, 0.1)  # select quantile here
targetval_2 = np.quantile(res2, 0.1)  # select quantile here
scoredf1 = pd.DataFrame(res1, columns=["O1"], index=embeddings.index)
scoredf2 = pd.DataFrame(res2, columns=["O2"], index=embeddings.index)
allvals = pd.concat([biked, scoredf1, scoredf2], axis=1)
allvals.dropna(axis=0, inplace=True)

_, y_struct, _, xscaler = load_data.load_framed_dataset("r", onehot=True, scaled=True, augmented=True)
y_struct = y_struct.loc[:, ["Sim 1 Safety Factor (Inverted)", "Model Mass Magnitude"]]
allvals.index = allvals.index.map(str)
commonidx = set(list(y_struct.index)).intersection(set(list(allvals.index)))

nunique = allvals.nunique()
cols_to_drop = nunique[nunique == 1].index
allvals = allvals.drop(cols_to_drop, axis=1)

allvals = pd.concat([allvals.loc[list(commonidx)], y_struct.loc[list(commonidx)]], axis=1)
# predictor1 = TabularPredictor(label="Model Mass Magnitude").fit(train_data=allvals.drop(["O1", "O2", "Sim 1 Safety Factor (Inverted)"], axis=1))
# predictor2 = TabularPredictor(label="Sim 1 Safety Factor (Inverted)").fit(train_data=allvals.drop(["O1", "O2", "Model Mass Magnitude"], axis=1))
predictor1 = TabularPredictor.load("AutogluonModels/ag-20231010_103824")
predictor2 = TabularPredictor.load("AutogluonModels/ag-20231010_103834")
y = allvals.iloc[:, -4:]
x = allvals.iloc[:, :-4]
print(x)
print(y)

thread_count = GA_Clip_utils.init_mp()
print(thread_count)


def get_scores(valid_idxs, meanvals, pop_size):
    scores = np.full((pop_size, 2), 2.0)
    for i in range(len(valid_idxs)):
        img_emb = meanvals[i]
        idx = valid_idxs[i]
        cos = torch.nn.CosineSimilarity()
        meanvals = meanvals.cpu()
        distance1 = cos(img_emb, target_embedding1.cpu())
        distance2 = cos(img_emb, target_embedding2.cpu())
        scores[idx, 0] = 1 - distance1.cpu().detach().numpy()
        scores[idx, 1] = 1 - distance2.cpu().detach().numpy()
    return scores


# def convertmats(x):
#     mats = x[["Material=Steel", "Material=Aluminum", "Material=Titanium"]].values
#     for i in range(len(x.index)):
#         if mats[i, 0]> mats[i, 1] and mats[i, 0]> mats[i, 2]:
#             mats[i,:] = (0.827145,-0.465079,-0.54407)
#         elif mats[i, 1]> mats[i, 2]:
#             mats[i,:] = (-1.208978,2.150174,-0.54407)
#         else:
#             mats[i,:] = (-1.208978,-0.465079,1.83800)
#         # mats[i,:] = (0.827145,-0.465079,-0.54407)
#     x.loc[:,["Material=Steel", "Material=Aluminum", "Material=Titanium"]] = mats
#     return x
class predictor_wrapper_class(object):
    def __init__(self, ref_columns, predictor1, predictor2):
        self.ref_columns = ref_columns
        self.predictor1 = predictor1
        self.predictor2 = predictor2

    def predict(self, x):
        x = pd.DataFrame(x, columns=self.ref_columns)
        # x = convertmats(x)
        valid_idxs, meanvals = GA_Clip_utils.get_mean_embedding(x, dataset, processor, model, thread_count,
                                                                num_views=10)
        scores = get_scores(valid_idxs, meanvals, len(x.index))
        scores2 = self.predictor2.predict(x)
        scores3 = self.predictor1.predict(x)
        scores = np.concatenate(
            [scores, np.expand_dims(scores2.values, axis=1), np.expand_dims(scores3.values, axis=1)], axis=1)
        scores = pd.DataFrame(scores, columns=y.columns)
        return scores


regressor = predictor_wrapper_class(x.columns, predictor1, predictor2)
