import webbrowser
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('demo.html')

if __name__ == "__main__":
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(port=3000)
