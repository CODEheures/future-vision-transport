from flask import Flask, render_template, redirect, url_for, request
from data import Data
import serializer
import requests

app = Flask(__name__)
data = Data()


def get_images():
    img_list = list(data.df.image[0:4])
    images = [serializer.toBase64(data.images_dir + name) for name in img_list]
    return images


@app.route('/shuffle')
def shuffle_data():
    global data
    data = Data()
    return redirect(url_for('home'), code=302)


@app.route('/')
def home():
    images = get_images()
    return render_template('home.html', images=images)


@app.route('/predict')
def predict():

    # Get image and true mask for idx
    idx = int(request.args.get('idx'))
    numpy_image = data.get_image(idx)

    # call api
    json = serializer.serialize(numpy_image)
    response = requests.post(
        "https://api-predict.codeheures.fr/",
        headers={
            'Content-type': 'application/json',
            'Accept': 'application/json'
        },
        json={"image": json})
    json_response = response.json()
    serialized_predited_mask = json_response["predicted_mask"]
    mask_predict = serializer.deserialize(serialized_predited_mask)

    # return results
    images = get_images()
    image = serializer.toBase64(data.images_dir + data.df.loc[idx, 'image'])
    mask_true = serializer.toBase64(data.masks_dir + data.df.loc[idx, 'mask'])
    mask_predict = serializer.numpy_tobase64(mask_predict)
    return render_template('predict.html',
                           images=images,
                           image=image,
                           mask_true=mask_true,
                           mask_predict=mask_predict)
