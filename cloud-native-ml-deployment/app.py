from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("mnist_model.keras")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "service": "mnist-model-api"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({
            "error": "Missing required field: image"
        }), 400

    image = np.array(data["image"])
    image = image.reshape(1, 784).astype("float32") / 255.0

    prediction = model.predict(image, verbose=0)
    predicted_class = int(np.argmax(prediction))

    return jsonify({
        "prediction": predicted_class
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

