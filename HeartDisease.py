import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
# from sklearn.metrics import confusion_matrix
# from keras import metrics
from PIL import Image
st.sidebar.subheader('Table of Contents')
st.sidebar.write('1. ','<a href=#prediction-of-existence-of-cardiovascular-disease-in-patients>Introduction</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#source>Source</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#exploratory-data-analysis>Exploratory data analysis</a>', unsafe_allow_html=True)
st.sidebar.write('4. ','<a href=#data-cleaning-and-data-preprocessing>Data cleaning and data preprocessing</a>', unsafe_allow_html=True)
st.sidebar.write('5. ','<a href=#machine-learning-prediction-models>Machine learning prediction models</a>', unsafe_allow_html=True)
st.sidebar.write('6. ','<a href=#conclusion>Conclusion</a>', unsafe_allow_html=True)
st.sidebar.write('7. ','<a href=#interactive-prediction-of-heart-disease>Interactive prediction of heart disease</a>', unsafe_allow_html=True)

st.title('Prediction of existence of cardiovascular disease in patients')

st.header('Goal of this case study')
st.subheader('The goal of this case study is to create and find the best performing model to predict whether a person has cardiovascular disease. ')

st.header('Definitions of the cardiovascular heart metrics present in this dataset')
st.write('The heart.csv file contains 912 observations with 12 attributes. Click on the expander to show all definitions of all metrics.')
with st.expander('Metric definitions'):
    st.subheader('Age')
    st.write('Age of the patient [years]')
    st.subheader('Sex')
    st.write('Sex of the patient [M: Male, F: Female]')
    st.subheader('ChestPainType')
    st.write('Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]')
    st.subheader('RestingBP')
    st.write('Resting blood pressure [mm Hg]')
    st.subheader('Cholesterol')
    st.write('Serum cholesterol [mm/dl]')
    st.subheader('FastingBS')
    st.write('Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]')
    st.subheader('RestingECG')
    st.write("Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]")
    st.subheader('MaxHR')
    st.write('Maximum heart rate achieved [Numeric value between 60 and 202]')
    st.subheader('ExerciseAngina')
    st.write('Exercise-induced angina [Y: Yes, N: No]')
    st.subheader('Oldpeak')
    st.write('Oldpeak = ST [Numeric value measured in depression]')
    st.subheader('ST_Slope')
    st.write('The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]')
    st.subheader('HeartDisease')
    st.write('Output class [1: heart disease, 0: Normal]')
    
st.header('Source')
st.write('This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:')
st.markdown("""
            <ul>
            <li>Cleveland: 303 observations</li>
            <li>Hungarian: 294 observations</li>
            <li>Switzerland: 123 observations</li>
            <li>Long Beach VA: 200 observations</li>
            <li>Stalog (Heart) Data Set: 270 observations</li>
            </ul>
            """, unsafe_allow_html = True)
st.write('Total: 1190 observations')
st.write('Duplicated: 272 observations')
st.write('Final dataset: 918 observations')
st.write('Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning Repository on the following link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/')

st.subheader('Citation')
st.write('fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.')


st.header('Exploratory data analysis')
st.write('This is the source dataset with 918 observations and 12 columns')
datacode = """import numpy as np
import pandas as pd
heart = pd.read_csv('heart.csv')
heart
"""
st.code(datacode)

heart = pd.read_csv('heart.csv')
explor = heart.copy()
explor['Sex'] = explor['Sex'].map({'M':1,'F':0})
explor['ExerciseAngina'] = explor['ExerciseAngina'].map({'Y':1,'N':0})
explor['ST_Slope'] = explor['ST_Slope'].map({'Up':2,'Flat':1,'Down':0})
explor['RestingECG'] = explor['RestingECG'].map({'Normal':0,'ST':1,'LVH':2})
explor['ChestPainType'] = explor['ChestPainType'].map({'TA':3,'ATA':2,'NAP':1,'ASY':0})

st.dataframe(explor)

st.write('Description of the dataset')
code2 = """heart.describe()
"""
st.code(code2)
st.dataframe(explor.describe())

plotcorr = px.imshow(explor.corr(),zmin=-1,zmax=1)
st.write('Correlation heatmap (r-values)')
st.plotly_chart(plotcorr)

st.write('Scatterplot tool, use selection drop down boxes on the left to select axis variables')
col1, col2 = st.columns([1, 2])
with col1:
    xi = st.selectbox('X axis variable', explor.columns)
    yi = st.selectbox('Y axis variable', explor.columns)

xo = explor[xi]
yo = explor[yi]

with col2:
    plot2 = px.scatter(explor,xo,yo,color='HeartDisease', width=500, opacity=0.15,trendline='ols')
    st.plotly_chart(plot2)

st.write('Boxplot tool, use selection drop down boxes on the left to selected continuous variable')
col5,col6 = st.columns([1,2])
with col5:
    xbi = st.selectbox('X axis variable', ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak'])
    
plot3 = px.box(explor,y=xbi,orientation='v',width=500)
with col6:
    st.plotly_chart(plot3)
st.header('Data cleaning and data preprocessing')

image1 = Image.open('starting.png')
image2 = Image.open('imputed.png')

st.write('This scatter matrix is a collection of every possible combination of scatter plot between every continuous variable, a lot of zeros can be seen from the distribution of ‘Cholesterol’. It is a discrete variable thus the large number of observations at 0 is suspicious. A cholesterol of 0 is not possible to achieve in real life either so I will be using KNN Imputation to impute the observations of cholesterol when it’s 0.    ')
st.image(image1, caption='Scatter matrix of continuous variables prior to preprocessing')
with st.expander('Preprocessing code'):
    with st.echo():  
        heart['Cholesterol'] = np.where(heart['Cholesterol']==0,np.nan, heart['Cholesterol'])#Replace 0s with nan
        dummied = pd.get_dummies(heart, drop_first=True)#Convert categorical variables to dummies
        numcolvec = ['Age', 'RestingBP','Cholesterol', 'MaxHR', 'Oldpeak']#List of continuous variables
        numcol = dummied[numcolvec]
        typecols = dummied[[col for col in dummied.columns if col not in numcolvec]]#List of dummies
        numcols = (numcol-numcol.mean())/numcol.std()#Standardizes numcols
        standardized = pd.concat([numcols, typecols],axis=1)#Concatenate numcols and typecols together
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=23)#Imputer object
        imputed = pd.DataFrame(imputer.fit_transform(standardized),columns = standardized.columns)#Impute the column cholesterol with nan as missing value

st.image(image2, caption='Scatter matrix of continuous variables after preprocessing')



st.header('Machine learning prediction models')
st.subheader('Logistics regression')
logcode = """from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
xtr, xte, ytr, yte = train_test_split(imputed.drop(['HeartDisease'],axis=1), imputed['HeartDisease'], test_size=0.1, random_state=42)
reg = LogisticRegression()
reg.fit(xtr, ytr)
ypred = reg.predict(xte)
accuracy = reg.score(xte,yte)
cm = metrics.confusion_matrix(yte, ypred)
print(f'Accuracy: {accuracy}')
print(cm)
"""

logresult = """'Accuracy: 0.8695652173913043'
array([[31,  7],
       [ 5, 49]], dtype=int64)
"""
with st.expander('Logistics regression code'):
    st.code(logcode)
st.code(logresult)

st.subheader('KNN classification')
knncode = """from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
xtr, xte, ytr, yte = train_test_split(imputed.drop(['HeartDisease'],axis=1), imputed['HeartDisease'], test_size=0.1, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=9, p=2)
neigh.fit(xtr,ytr)
ypred = neigh.predict(xte)
accuracy = neigh.score(xte,yte)
cm = metrics.confusion_matrix(yte, ypred)
print(f'Accuracy: {accuracy}')
print(cm)
"""

knnresult = """'Accuracy: 0.8586956521739131'
array([[31,  8],
       [ 5, 48]], dtype=int64)
"""
with st.expander('KNN code'):
    st.code(knncode)
st.code(knnresult)

st.subheader('Neural network')
nncode = """import tensorflow as tf
from keras import metrics
ranstate = 51
xtr, xte, ytr, yte = train_test_split(imputed.drop(['HeartDisease'],axis=1), imputed['HeartDisease'], test_size=0.1, random_state=ranstate)
xtr, xval, ytr, yval = train_test_split(xtr, ytr, test_size=0.1, random_state=ranstate)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(15,)), # input layer
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax') # output layer
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NUM_EPOCHS = 50
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',verbose=1,baseline=0.86,patience=40)

model.fit(xtr.to_numpy(),ytr.to_numpy(), epochs=NUM_EPOCHS, callbacks=[earlystopping], validation_data=(xval.to_numpy(), yval.to_numpy()), verbose =2)
test_loss, test_accuracy = model.evaluate(xte,yte)
cm = confusion_matrix(yte.to_numpy(), ypred)
print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')
print(cm)
"""
nnresult = """'Loss: 0.3108227849006653, Accuracy: 0.8804348111152649'
array([[27,  6],
       [ 5, 54]], dtype=int64)
"""
with st.expander('Neural network code'):
    st.code(nncode)
st.code(nnresult)

st.header('Conclusion')
accuracies = pd.DataFrame()
accuracies['Models'] = ['Logistics regression','KNN classification','Neural Network']
accuracies['Accuracy'] = [0.87,0.86,0.88]
graph = px.bar(accuracies,'Accuracy', 'Models',  range_x=[0.5,1], orientation='h', height=250)
st.plotly_chart(graph)
st.write('The neural network performed the best by only a small margin, however I would use the logistics regression because of its simpleness and speed. The neural network had some slight overfitting issues due to correlations within the imputed data. ')

st.header('Interactive prediction of heart disease')
st.write('A simple way to utilise the prediction models is to enter values into the form below, click on the form submit button, and it will update the prediction to be whether a patient with the entered values will be predicted as having cardiovascular problems or not. Default values are the mean, range is from the minimum value the variable ever appeared at to maximum value the variable ever appeared at.')
import pickle
def save_pkl(python_object,file):
    my_pkl = open(file,'wb')
    pickle.dump(python_object,my_pkl)
    my_pkl.close()

def load_pkl(file):
    my_pkl=open(file,'rb')
    python_object = pickle.load(my_pkl)
    my_pkl.close()
    return(python_object)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

logregmodel = load_pkl('heartlogregmodel.pkl')
knnmodel = load_pkl('heartknnmodel.pkl')
import keras.models
nnmodel = keras.models.load_model('actualmodel')
standinfo = load_pkl('standinfo.pkl')
col = heart.drop('HeartDisease',axis=1).columns
icol = imputed.drop('HeartDisease',axis=1).columns
exampledf = pd.DataFrame(np.array([[0 for i in range(11)]]),columns=col)
exampledf['Sex'] = 'M/F'
exampledf['ChestPainType'] = 'ATA/NAP/ASY/TA'
exampledf['FastingBS'] = '1 for above 120mg/dl, 0 for below'
exampledf['RestingECG'] = 'Normal/ST/LVH'
exampledf['ExerciseAngina'] = 'Y/N'
exampledf['ST_Slope'] = 'Up/Flat/Down'
with st.form('form1'):
    col3, col4 = st.columns(2)
    with col3:#[0,3,4,7,9]
        ni1 = st.number_input(col[0],key='a',value=standinfo[col[0]][4], min_value=standinfo[col[0]][2], max_value=standinfo[col[0]][3], step=1.)
        ni2 = st.number_input(col[3],key='b',value=standinfo[col[3]][4], min_value=standinfo[col[3]][2], max_value=standinfo[col[3]][3], step=1.)
        ni3 = st.number_input(col[4],key='c',value=standinfo[col[4]][4], min_value=standinfo[col[4]][2], max_value=standinfo[col[4]][3], step=1.)
        ni4 = st.number_input(col[7],key='d',value=standinfo[col[7]][4], min_value=standinfo[col[7]][2], max_value=standinfo[col[7]][3], step=1.)
        ni5 = st.number_input(col[9],key='e',value=standinfo[col[9]][4], min_value=standinfo[col[9]][2], max_value=standinfo[col[9]][3], step=0.1)
    with col4:#[1,2,5,6,8,10]
        si1 = st.selectbox(col[1],['M','F'],key='f')
        si2 = st.selectbox(col[2],['ATA','NAP','ASY','TA'],key='g')
        si3 = st.selectbox(col[5],['Above 120mg/dl','Below 120mg/dl'],key='h')
        si4 = st.selectbox(col[6],['Normal','ST','LVH'],key='i')
        si5 = st.selectbox(col[8],['Yes','No'],key='j')
        si6 = st.selectbox(col[10],['Up','Flat','Down'],key='k')
    
    nnsubmitted = st.form_submit_button("Submit to neural network model")
    logsubmitted = st.form_submit_button("Submit to logistics regression model")
    knnsubmitted = st.form_submit_button("Submit to KNN model")
    if nnsubmitted or logsubmitted or knnsubmitted:
        
        inputdf = pd.DataFrame(np.array([[0.0 for i in range(15)]]), columns=icol)
        inputdf['Age'] = (ni1-standinfo['Age'][0])/standinfo['Age'][1]
        inputdf['RestingBP'] = (ni2-standinfo['RestingBP'][0])/standinfo['RestingBP'][1]
        inputdf['Cholesterol'] = (ni3-standinfo['Cholesterol'][0])/standinfo['Cholesterol'][1]
        inputdf['MaxHR'] = (ni4-standinfo['MaxHR'][0])/standinfo['MaxHR'][1]
        inputdf['Oldpeak'] = (ni5-standinfo['Oldpeak'][0])/standinfo['Oldpeak'][1]
        if si1=='M':
            inputdf['Sex_M'] = 1
        elif si1=='F':
            inputdf['Sex_M'] = 0
        if si2=='ATA':
            inputdf['ChestPainType_ATA'] = 1
        elif si2=='NAP':
            inputdf['ChestPainType_NAP'] = 1
        elif si2=='ASY':
            pass
        elif si2=='TA':
          inputdf['ChestPainType_TA'] = 1  
        if si3=='Above 120mg/dl':
            inputdf['FastingBS'] = 1
        elif si3=='Below 120mg/dl':
            pass
        if si4=='Normal':
            inputdf['RestingECG_Normal'] = 1
        elif si4=='ST':
            inputdf['RestingECG_ST'] = 1
        elif si4=='LVH':
            pass
        if si5=='Yes':
            inputdf['ExerciseAngina_Y'] = 1
        elif si5=='No':
            pass
        if si6=='Up':
            inputdf['ST_Slope_Up'] = 1
        elif si6=='Flat':
            inputdf['ST_Slope_Flat'] = 1
        elif si6=='Down':
            pass
        inputdf
        if nnsubmitted:
            ypredprob = nnmodel.predict(inputdf.to_numpy())
            ypred = np.where(ypredprob[:,1]<0.5, 0, 1)
            result = int(ypred[0])
            st.write(f'Person with above attributes is predicted to {np.where(result>0.5, "have", "not have")} cardiovascular problems({result})')
        if logsubmitted:
            ypred = logregmodel.predict(inputdf.to_numpy())
            result = int(ypred[0])
            st.write(f'Person with above attributes is predicted to {np.where(result>0.5, "have", "not have")} cardiovascular problems({result})')
        if knnsubmitted:
            ypred = knnmodel.predict(inputdf.to_numpy())
            result = int(ypred[0])
            st.write(f'Person with above attributes is predicted to {np.where(result>0.5, "have", "not have")} cardiovascular problems({result})')

            
# st.write('Alternatively, you can choose to upload a file of observations to predict, note that column headings and content must be in the order and names below, scroll horizontally to view full labels')
# st.dataframe(exampledf)
# with st.form('form2'):
#     thefile = st.file_uploader('Upload csv file for prediction', type=['csv'])
#     userfile = pd.read_csv(thefile)
#     filedummied = pd.get_dummies(userfile, drop_first=True)
#     filenumcol = dummied[numcolvec]
#     filenumcols = (filenumcols-numcol.mean())/numcol.std()