import pickle
import pandas
import numpy as np
from sklearn.preprocessing  import OrdinalEncoder

class aaskAI:
    def __init__(self, aipath, encpath):
        with open(aipath, 'rb') as file:
            self.model = pickle.load(file)
        with open(encpath, 'rb') as file:
            self.encodes = pickle.load(file)

    def preprocess(self,data):
        def toBinary(obj,s1,s2):
            return obj.str.replace(s1,'0').str.replace(s2,'1').astype(np.int64)
        def toAvg(obj):
            return obj.str.split('-').apply(lambda x: (float(x[0]) + float(x[1]))/2)
        data['Age'] = toAvg(data['Age'])
        data['Study Hours'] = toAvg(data['Study Hours'])
        data['GPA Last'] = toAvg(data['GPA Last'])
        data['Sex'] = toBinary(data['Sex'],'M','F')
        data['Marital Status'] = toBinary(data['Marital Status'], 'Unmarried', 'Married')
        data['Scholarship'] = data['Scholarship'].str.split('%').str.get(0).astype(np.int64)

        orderEDU = [['Primary', 'Secondary', 'High', 'University', 'MSc.', 'Ph.D.']]
        orderNSO = [['Never','Sometimes','Often']]
        orderNNP = [['Negative','Neutral','Positive']]
        orderNBR = [['Never', 'Before Exam', 'Regular']]
        orderNSA = [['Never', 'Sometimes', 'Always']]
        catmaps = {'Mother\'s Education':orderEDU,'Father\'s Education':orderEDU,'Reading NonSci':orderNSO, 'Reading Sci':orderNSO, 'Project Impact':orderNNP, 'Midterm Prep - 2':orderNBR,'Notes Taken':orderNSA, 'Listening':orderNSA, 'Discussion Interest':orderNSA}

        for cats in catmaps:
            encode = OrdinalEncoder(categories = catmaps[cats])
            data[cats] = encode.fit_transform(data[[cats]]).astype(np.int64)

        cat_cols = data.select_dtypes(include=[object]).columns
        for cats in cat_cols:
            data[cats] = self.encodes[cats].fit_transform(data[cats]).astype(np.int64)


        data['Study Habits'] = data[['Reading Sci','Reading NonSci','Study Hours', 'Midterm Prep - 2', 'Notes Taken','Listening']].mean(axis=1)
        data['Study Impact'] = data[['Project Impact','Discussion Interest']].max(axis=1)
        data['mid']=data[['Mother\'s Occupation','Father\'s Occupation','Income','Mother\'s Education','Father\'s Education']].max(axis=1)
        data['Family Support']=data[['mid','Parental Status']].mean(axis=1)


        return data[['Age','Sex','Marital Status','Scholarship','Study Impact','Study Habits','Family Support','GPA Last','COURSE ID']]

    def predict(self, entry):
        return self.model.predict(entry)
