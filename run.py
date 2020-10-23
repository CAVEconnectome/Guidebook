from werkzeug.serving import WSGIRequestHandler
from guidebook.app import create_app
import os

HOME = os.path.expanduser("~")

app = create_app()

if __name__ == "__main__":
    # WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0',
            port=4001,
            debug=True,
            threaded=True)
