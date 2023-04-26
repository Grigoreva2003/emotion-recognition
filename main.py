import sys, json, os

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_flatpages import FlatPages, pygments_style_defs
from flask_frozen import Freezer
from werkzeug.utils import secure_filename
from django.utils.cache import add_never_cache_headers

from image_processing import process_image


class NoCachingMiddleware(object):
    def process_response(self, request, response):
        add_never_cache_headers(response)
        return response


DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FLATPAGES_ROOT = 'content'
POST_DIR = 'posts'

UPLOAD_FOLDER = 'static\img\IO_img'
ALLOWED_EXTENSIONS = {'png', 'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__)
flatpages = FlatPages(app)
freezer = Freezer(app)
app.config.from_object(__name__)

with open('settings.txt', encoding='utf8') as config:
    data = config.read()
    settings = json.loads(data)


# home page of the site
@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('index.html', **settings)


# method receives and saves IO_img image from the submission form
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['photo']
        if file:
            filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))

    return render_template('loading_form.html', filename='', **settings)


# method displays the received file
@app.route('/upload/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    if request.method == 'POST':
        file = request.files['photo']
        if file:
            filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))

    filename = 'http://127.0.0.1:8000/uploads/' + filename
    return render_template('loading_form.html', filename=filename, **settings)


# utility method for loading image in html
@app.route('/uploads/<filename>')
def send_file(filename):
    output_file = process_image(UPLOAD_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, output_file)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(host='127.0.0.1', port=8000, debug=True)
