import numpy as np 
import matplotlib.pyplot as plt
import plotly as p 
import seaborn as sns
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

import plotly.express as px 


df = pd.read_csv("C:\Program Files\student_learning_dataset.csv") 

pd.set_option('display.max_columns', None)  # Show all columns


print(df.head()) #see all the data in the dataset


print(df.describe())



#=======BINIRIZATION==========

