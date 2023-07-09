from flask import Flask, request, make_response
from PIL import Image
import io
import base64
import numpy as np

hair = ["blonde hair", "brown hair", "black hair", "blue hair", "pink hair", "purple hair", "green hair", "red hair", "silver hair", "white hair", "orange hair", "aqua hair", "grey hair"]
eyes = ["blue eyes", "red eyes", "brown eyes", "green eyes", "purple eyes", "yellow eyes", "pink eyes", "aqua eyes", "black eyes", "orange eyes"]
userMappingVector = {}


def generatePoints(batch_size, dim):
  hair_selector = np.random.randint(0,13, size = (batch_size))
  hair_indices = np.arange(0,batch_size)
  hair_features = np.zeros((batch_size,13))
  hair_features[hair_indices,hair_selector] = 1
  eye_selector = np.random.randint(0,10, size = (batch_size))
  eye_indices = np.arange(0,batch_size)
  eye_features = np.zeros((batch_size,10))
  eye_features[eye_indices,eye_selector] = 1
  features = np.random.randn(batch_size,dim)
  labels = np.concatenate((hair_features,eye_features), axis = 1)
  #finalFeatures = np.concatenate((features,labels), axis = 1)
  labels = labels
  finalFeatures = features
  return finalFeatures, labels

def generateOneHotHairFeatures(hairC, random = False):
    if random:
        hair_selector = np.random.randint(0,13, size = (1))
        hair_indices = np.arange(0,1)
        hair_features = np.zeros((1,13))
        hair_features[hair_indices,hair_selector] = 1
        return hair_features
    hair_features = np.zeros((1,13))
    hair_features[0,hair.index(hairC)] = 1
    return hair_features

def generateOneHotEyeFeatures(eyesC, random = False):
    if random:
        eye_selector = np.random.randint(0,10, size = (1))
        eye_indices = np.arange(0,1)
        eye_features = np.zeros((1,10))
        eye_features[eye_indices,eye_selector] = 1
        return eye_features
    eye_features = np.zeros((1,10))
    eye_features[0,eyes.index(eyesC)] = 1
    return eye_features


def create_app(generatorFunctions):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.static_folder = 'static'    

    from.views import views
    from.auth import auth

    app.register_blueprint(views, url_prefix='/')        
    app.register_blueprint(auth, url_prefix='/')
    @app.route('/generate-image', methods=['POST'])
    def generate_image():
        vals = request.form
        toUse = vals['model']
        hairC = vals['hair color']
        eyesC = vals['eye color']
        index = 0
        if toUse == "KanonNet Original":
            inputValues, labels = generatePoints(1,128)
            images = generatorFunctions[index](inputValues).numpy().transpose(0, 2, 3, 1)
        else:
            if toUse == "KanonNet Colored Alpha":
                index = 1 
            elif toUse == "KanonNet Colored Beta":
                index = 2 
            elif toUse == "KanonNet Colored Mobile":
                index = 3 
            hair_features = generateOneHotHairFeatures(hairC, hairC == 'Random')
            eye_features = generateOneHotEyeFeatures(eyesC, eyesC == 'Random')
            inputValues, labels = generatePoints(1,128)
            if vals['userId'] in userMappingVector and vals['noise'] == 'fixed':
                inputValues = userMappingVector[vals['userId']]
            else:
                userMappingVector[vals['userId']] = inputValues
            labels = np.concatenate((hair_features,eye_features), axis = 1)
            images = generatorFunctions[index](inputValues, labels).numpy().transpose(0, 2, 3, 1)
        images = (images + 1)/2
        np_array = images[0]*255
        np_array = np_array.astype(np.uint8)
        # Convert the NumPy array to PIL image
        image = Image.fromarray(np_array)
        image = image.resize((512,512))
        # Convert the image to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)
        image_bytes = img_io.getvalue()

        # Convert the image bytes to base64-encoded string
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Set the response MIME type
        response = make_response(image_base64)
        response.mimetype = 'text/plain'

        return response                                                                                                                         
    return app