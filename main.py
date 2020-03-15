import os
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
	return render_template('templates/index.html')


@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)

if __name__ == '__main__':
    app.run()