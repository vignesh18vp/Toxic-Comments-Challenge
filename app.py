from flask import Flask, render_template,url_for,request
import numpy as np
import pandas as pd
import re
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
ps = PorterStemmer()

Commentdataset = pd.read_csv("data/train.csv")
modelfile = open("data/NB_Model_lat.pkl",'rb')
model = pickle.load(modelfile)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html",tabIndicator='HomePage')

@app.route('/Index')
def Index():
    return render_template("home.html",tabIndicator='HomePage')
    
@app.route('/DataAnalysis',methods = ['GET'])
def DataAnalysis():    
    dataAnalysis, TotalRecordCount,CleanCommentRecords = ExploratoryDataAnalysis(Commentdataset)
    return render_template("DataAnalysis.html",
                           tabIndicator='DAPage',
                           OriginalDataset = Commentdataset.head(20).to_html(classes='table table-sm table-striped  table-bordered'), 
                           AnalysisData = dataAnalysis.to_html(header = False,classes='table table-sm table-striped  table-bordered'),
                           TotalRec = TotalRecordCount,
                           CleanRecords = CleanCommentRecords)

@app.route('/CommentsRetrieve',methods = ['POST'])
def CommentsRetrieve(): 
    category = request.form['CommentBtnClick']
    filterCategory = ""
    if category == 'Toxic':
        filterCategory = "toxic"
    elif category == 'Severe Toxic':
        filterCategory = "severe_toxic"
    elif category == 'Obscene':
        filterCategory = "obscene"
    elif category == 'Threat':
        filterCategory = "threat"
    elif category == 'Insult':
        filterCategory = "insult"  
    else:
        filterCategory = "identity_hate"

    dataAnalysis, TotalRecordCount,CleanCommentRecords = ExploratoryDataAnalysis(Commentdataset)
    return render_template("DataAnalysis.html",
                           tabIndicator='DAPage',
                           OriginalDataset = Commentdataset[Commentdataset[filterCategory] == 1].head(20).to_html(classes='table table-sm table-striped  table-bordered'), 
                           AnalysisData = dataAnalysis.to_html(header = False,classes='table table-sm table-striped  table-bordered'),
                           TotalRec = TotalRecordCount,
                           CleanRecords = CleanCommentRecords)


@app.route('/DataVisualize')
def DataVisualize():
    return render_template("DataVisualize.html",tabIndicator='VisPage')


@app.route('/DataProcessing', methods=['GET','POST'])
def DataProcessing():
    Lemma_text = ""
    SymbolsRemoved = ""
    LowercaseText = ""
    OrginalComment = ""
    if request.method == 'POST':
        OrginalComment = request.form['CommentText']
        if(OrginalComment != ""):
            Lemma_text,SymbolsRemoved,LowercaseText = GetCleanedData(OrginalComment)
    return render_template("DataProcessing.html",tabIndicator='PrePage',OriginalText = OrginalComment, SymbolsText = SymbolsRemoved,LowerText = LowercaseText,LemmaText = Lemma_text)



@app.route('/NBModel')
def NBModel():
    return render_template("NBModel.html",tabIndicator='NBPage')


@app.route('/NBModelPredict',methods = ['GET','POST'])
def NBModelPredict():
    Processedcomment = ""
    OrginalComment = ""
    prediction = []
    detailedPred = []
    Errormsg = ""
    if request.method == 'POST':        
        OrginalComment = request.form['CommentText']
        if(OrginalComment != ""):
            Processedcomment,SymbolsRemoved,LowercaseText = GetCleanedData(OrginalComment)
            if(Processedcomment != ""):                
                prediction, detailedPred = Predict_NB(Processedcomment)
            else:
                Errormsg = "Not a Valid Comment Text"
    return render_template("NBModelPredict.html",tabIndicator='NBPPage',Error=Errormsg, OriginalText = OrginalComment,ProcessedText = Processedcomment, output = prediction, detailedOutput = detailedPred)



@app.route('/LSTMModel')
def LSTMModel():
    return render_template("LSTMModel.html",tabIndicator='LSTMPage')

@app.route('/LSTMModelPredict',methods = ['GET','POST'])
def LSTMModelPredict():
    Processedcomment = ""
    OrginalComment = ""
    prediction = []
    detailedPred = []
    Errormsg = ""
    if request.method == 'POST':        
        OrginalComment = request.form['CommentText']
        if(OrginalComment != ""):
            Processedcomment,SymbolsRemoved,LowercaseText = GetCleanedData(OrginalComment)
            if(Processedcomment != ""):                
                prediction, detailedPred = Predict_LSTM(Processedcomment)
            else:
                Errormsg = "Not a Valid Comment Text"
    return render_template("LSTMModelPredict.html",tabIndicator='LSTMPPage',Error=Errormsg, OriginalText = OrginalComment,ProcessedText = Processedcomment, output = prediction, detailedOutput = detailedPred)


def ExploratoryDataAnalysis(dataset):           
        ColumnsList = []
        NullList = []
        NAList = []
        CountList = []    
        
        for col in dataset.columns:        
            if(col == 'id' or col == 'comment_text'):
                continue
            CountList.append(dataset[dataset[col] == 1].shape[0])
            ColumnsList.append(col)
            NullList.append(dataset[dataset[col] == 1]["comment_text"].isnull().sum())
            NAList.append(dataset[dataset[col] == 1]["comment_text"].isna().sum())
        
        cleancomment = dataset[(dataset.toxic == 0) & (dataset.severe_toxic == 0) & (dataset.obscene == 0) &  (dataset.threat == 0) &  (dataset.insult == 0)  &  (dataset.identity_hate == 0)].shape[0]
        df = pd.DataFrame(list(zip(ColumnsList,NullList,NAList,CountList)), columns =['Column Name','Null Count','NA Count','Record Count']) 
        return df.transpose(),dataset.shape[0],cleancomment
    
def Predict_NB(comment):
  categories = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
  probs = model.predict_proba([comment])[0]
  result = []
  detailedResult = []
  for (prob, category) in zip(probs, categories):
      detailedResult.append(category +':'+ str(round(prob,2)*100)+'%')
      if(round(prob,2)*100 > 55):
          result.append(category +':'+ str(round(prob,2)*100)+'%')      
  if(len(result) == 0):
      result.append("Clean Comment")
  return result , detailedResult     



def Predict_LSTM(comment):        
    from keras.preprocessing import text, sequence
    from keras.preprocessing.sequence import pad_sequences
    from keras import backend as K
    modelfileDL = open("data/DL_Keras_Model.pkl",'rb')
    modelDL = pickle.load(modelfileDL)
    modelDL._make_predict_function()
    categories = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    tokenizer = text.Tokenizer(num_words=100000)
    seq = tokenizer.texts_to_sequences([comment])
    comment_padded = pad_sequences(seq, maxlen=150)
    DLResult = modelDL.predict(comment_padded)
    K.clear_session()
    result = []
    detailedResult = []
    temp = []
    for i in range (0, 6):
      temp.append(DLResult[0][i])

    for (prob, category) in zip(temp, categories):
      detailedResult.append(category +':'+ str(prob))
      if(category == 'Toxic' and prob > 0.05):
          result.append(category +':'+ str(prob))
      elif(prob > 0.005):
          result.append(category +':'+ str(prob))      
    
    if(len(result) == 0):
      result.append("Clean Comment")
  
    return result , detailedResult  


def GetCleanedData(comment_text):
    stop_words=set(stopwords.words("english"))    
    Remove_words = [ 'no', 'nor', 'not','don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    for item in Remove_words: 
        stop_words.remove(item) 
    
    # remove non alphabatic characters
    SymbolsRemoved = re.sub('[^A-Za-z]', ' ', comment_text)
    # make words lowercase, because Go and go will be considered as two words
    LowercaseText = SymbolsRemoved.lower()
    # tokenising
    tokenized_comments = word_tokenize(LowercaseText)
    
    #Remvoing stop words and Stemming
    comment_processed = []
    for item in tokenized_comments:
        if item not in (stop_words):
            comment_processed.append(ps.stem(item))
    comment_text = " ".join(comment_processed)
    return comment_text,SymbolsRemoved,LowercaseText


if(__name__ == "__main__"):
    app.run(debug=True)

