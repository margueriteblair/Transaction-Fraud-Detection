from flask import Flask, jsonify
import analytics

app = Flask(__name__)

@app.route("/api", methods=['GET'])
def index():
    return "this is a test! successfully launched py server"

@app.route("/api/fraud-det", methods=['GET'])
def fetch_model():
    #jsonify function in flask returns a flask.Response() object that already has the
    #appropriate content-type header 'application/json' for
    analysis = analytics.train_model()
    return jsonify(analysis)

if __name__ == "__main__":
    app.run(host='localhost', port=9999)

#currently following this tutorial: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
