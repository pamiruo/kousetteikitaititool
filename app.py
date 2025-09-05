from flask import Flask, render_template
from logic.tokyoguru import tokyoguru_handler
from logic.magireko import magireko_handler
from logic.monkeyturn5 import monkeyturn5_handler
from logic.karakuri import karakuri_handler

app = Flask(__name__)

# トップページ（とりあえず東京喰種にリダイレクト or 簡単なリンクでも可）
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 東京喰種ルート
@app.route("/tokyoguru", methods=["GET", "POST"])
def tokyoguru():
    return tokyoguru_handler()

# マギレコルート
@app.route("/magireko", methods=["GET", "POST"])
def magireko():
    return magireko_handler()

# モンキールート
@app.route("/monkeyturn5", methods=["GET", "POST"])
def monkeyturn5():
    return monkeyturn5_handler()

# からくりルート
@app.route("/karakuri", methods=["GET", "POST"])
def karakuri():
    return karakuri_handler()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)


# python -m venv venv
# .\venv\Scripts\activate