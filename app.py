import pickle
from flask import Flask, render_template, request

import cv2
import numpy as np

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# #from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

app=Flask(__name__)

with open("brain_tumor_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    
    img=cv2.imread(image_path,0)
    img=cv2.resize(img,(200,200))
    z=[]
    z.append(img)
    z=np.array(z)
    z_u= z.reshape(len(z), -1)
    z_u=z_u/255
    
    

   

    

    


    yhat=loaded_model.predict(z_u)

    if yhat==0:
        # classification = '%s' % (yhat)
        classification = 'no tumor' 
    elif yhat==1:
        classification = 'pituitary tumor'
    elif yhat==2:
        classification='glioma tumor'
    elif yhat==3:
        classification='meningioma tumor'

    


    return render_template('index.html', prediction=classification)


if __name__=='__main__':
    app.run(port=3000, debug=True)