"""
NoghenOMR - Optical Music Recognition Backend
Flask server for processing scanned sheet music PDFs
"""

from flask import Flask
from flask_cors import CORS
from server.omr import omr_bp

app = Flask(__name__)
CORS(app)

# Register the OMR blueprint
app.register_blueprint(omr_bp)


@app.route('/health')
def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    print("Starting NoghenOMR backend on http://localhost:4000")
    app.run(host='0.0.0.0', port=4000, debug=True)
