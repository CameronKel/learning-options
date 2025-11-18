from flask import Flask, redirect, url_for
from greeks import bp as greeks_bp

app = Flask(__name__)

# Register the Greeks blueprint
app.register_blueprint(greeks_bp)

@app.route("/")
def index():
    # Redirect root to the Greeks page
    return redirect(url_for("greeks.greeks"))

if __name__ == "__main__":
    app.run(debug=True)
