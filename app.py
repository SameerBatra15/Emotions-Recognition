from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os,cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
app = Flask(__name__)

model=keras.models.load_model("expression.h5")
class_to_label = {0 :'ANGRY', 1 : 'DISGUST', 2:'FEAR', 3 :'HAPPY', 4:'SAD', 5:'SURPRISE', 6:'NEUTRAL'}

def model_predict(img_path, model):
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img,1.3,5)
    x,y,w,h = faces[0]
    imgg=img[y:y+h,x:x+h]
    imgg=cv2.resize(imgg,(48,48))/255.0
    imgg=imgg.reshape(1,48,48,1)

    n_pred=model.predict_classes(imgg)
    output=class_to_label[n_pred[0]]
    return output



@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        output=model_predict(file_path,model)
        

        return render_template('pred.html',output=output,file_name=str(f.filename))
	
@app.route('/upload/<filename>')
def upload_img(filename):
    return send_from_directory("uploads", filename)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)