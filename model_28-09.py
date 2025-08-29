from settings import *


#Loading the data
df = pd.read_csv(r"students_randomized.csv")
df = pd.DataFrame(df)

#CLEANING THE DATA AND 
data = df.drop(columns = ['Student_ID','Student_Name'])
data = pd.DataFrame(data)


# = = = = = = = = = = PREPROCESSING = = = = = = = = = = = = = 
#Label Encoding
input_labels =['Diagrams and Charts','Color Coding','Spatial Awareness','Lecture and Discussions',
'Repetition and Rhythms','Explaining to others','Text Notes','List and Outlines',
'Research and Reports','Hands-on Practice','Movement Memory','Real-life Application',
'Patterns and Sequences','Problem solving','Data and Numbers','Teamwork','Peer Feedback',
'Role Play','Independent Study','Self-Reflection','Goal Setting','Environment Connection',
'Classification','Observation','Rhythm Memory','Background Music','Song recreation']

#create label Encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

le = LabelEncoder()
data['Educational_Strategy'] = encoder.fit_transform(df['Educational_Strategy'])



#SCALING
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(data)
#print ("\nMin max scaled data:\n", data_scaled_minmax)


data = data.to_numpy()



#NAIVE BAYES --- GAUSSIAN
X = data_scaled_minmax[:, :-1]  
y = data[:, -1]

print(y)


#Split into 3 bins: Low, Medium, High
bins = [20, 40, 60, 80]
y = np.digitize(y, bins)-1

print(y)

test_size = random.randint(1,100)
test_size =30
test_size = test_size/100

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 42)





gnb = GaussianNB()
model = gnb.fit(X_train, y_train)

preds = gnb.predict(X_test)

print(preds)


print(accuracy_score(y_test,preds)*100 , '%')







