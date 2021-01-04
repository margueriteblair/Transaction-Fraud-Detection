from flask import Flask, jsonify
from flask_restful import Api
import analytics

app = Flask(__name__)
api = Api(app)

@app.route("/api")
def index():
    return "this is a test! successfully launched py server"

@app.route("/api/fraud-det")
def fetch_model():
    result = analytics.train_model()
    return (jsonify(result), 200)

if __name__ == "__main__":
    app.run(host='localhost', port=9999)

#currently following this tutorial: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
