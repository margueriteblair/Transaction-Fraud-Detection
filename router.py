from flask import Flask, jsonify
from flask_restful import Api

app = Flask(__name__)
api = Api(app)

#currently following this tutorial: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
