from flask import Flask, render_template
from data_analysis import generate_charts_for_web
from pathlib import Path


app = Flask(__name__)

@app.route("/")
def home():
 return render_template("index.html")
if __name__ == "__main__": 
    app.run(debug=True)