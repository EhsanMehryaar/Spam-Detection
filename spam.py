import pandas as pd
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble 
from sklearn.metrics import classification_report, accuracy_score


def df_maker(name):
    ''' 
        Utility function that imports the data and creates dataframe and deletes duplicate entries. 
    '''
    # Read the CSV file
    df = pd.read_csv(name)
    df['spam'] = df['spam'].astype(int)

    # Remove duplicates
    df = df.drop_duplicates()

    df = df.reset_index(inplace = False)[['text','spam']]
    return df

def clean (df):
    ''' 
        Utility function changes all the letter to lower case, removes punctuation, tags, digits and 
        special characters. 
    '''
    cleaned = []
    for w in range(len(df.text)):
        #Lower case
        temp = df['text'][w].lower()
    
        #Delete punctuation
        temp = re.sub('[^a-zA-Z]', ' ', temp)
    
        #Delete tags
        temp = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",temp)
    
        #Delete digits and special charachters
        temp = re.sub("(\\d|\\W)+"," ",temp)
    
        cleaned.append(temp)

    df['text'] = cleaned
    return df

def main(): 
    # Creating dataframe
    df = df_maker('emails.csv')

    #Cleaning data
    df = clean(df)

    vecorized_text = CountVectorizer().fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(vecorized_text, df['spam'], test_size = 0.45, random_state = 42, shuffle = True)

    classifier = ensemble.GradientBoostingClassifier(
        n_estimators = 100, #number of decision trees
        learning_rate = 0.5, #learning rate
        max_depth = 6
    )
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred))


if __name__ == "__main__": 
    # calling main function 
    main()

