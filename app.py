from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import joblib
import pickle


# importing model

model = joblib.load(open('model.pkl','rb'))
sc = joblib.load(open('standscaler.pkl','rb'))
ms = joblib.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    print(N,P,K,temp,humidity,ph,rainfall)
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    flag = 0
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        flag=1
        result = "{} is the best crop to be cultivated right there \n".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    if(flag==1):
        return render_template('index.html',result = result,N=N,P=P,K=K,temp=temp,humidity=humidity,ph=ph,rainfall=rainfall)
    else:
        return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)