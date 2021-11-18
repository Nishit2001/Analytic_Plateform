import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask, render_template, request, session, redirect
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config['uploader'] = "E:\\DM-project"


@app.route('/')
def home():
    return render_template('layout.html')


@app.route('/main')
def main():
    dataset = pd.read_csv(f)

    dataset['experience'].fillna(0, inplace=True)

    dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

    X = dataset.iloc[:, :3]

    # Converting words to integer values
    # def convert_to_int(word):
    # word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
    # 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    # return word_dict[word]

    # X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

    y = dataset.iloc[:, -1]

    # Splitting Training and Test Set
    # Since we have a very small dataset, we will train our model with all availabe data.

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()

    # Fitting model with trainig data
    regressor.fit(X, y)

    # Saving model to disk
    pickle.dump(regressor, open('model.pkl', 'wb'))

    # Loading model to compare the results
    model = pickle.load(open('model.pkl', 'rb'))
    print(model.predict([[2, 9, 6]]))
    return render_template('index1.html')


@app.route('/main1')
def main1():
    dataset = pd.read_csv(f)
    X = dataset.iloc[:, :3]
    y = dataset.iloc[:, -1]

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)

    # Saving model to disk
    pickle.dump(classifier, open('model.pkl', 'wb'))

    # Loading model to compare the results
    model = pickle.load(open('model.pkl', 'rb'))
    print(model.predict([[2, 9, 6]]))
    return render_template('index1.html')

@app.route('/main2')
def main2():
    dataset = pd.read_csv(f)
    X = dataset.iloc[:, :3]
    #y = dataset.iloc[:, -1]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    mo = kmeans.fit(X)
    pickle.dump(mo, open('model.pkl','wb'))
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if os.path.exists("data.csv"):
        os.remove("data.csv")

    return render_template('index1.html', prediction_text='Predicted attribute is ${}'.format(output))

@app.route("/upload", methods=['POST'])
def upload():
    global f
    if request.method == "POST":
        file = request.files["file"]
        f = file.filename
        file.save(os.path.join(app.config["uploader"], secure_filename(file.filename)))
        return render_template("options.html", message="File uploaded successfully!")



@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
