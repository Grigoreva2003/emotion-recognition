import sys, json, os

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_flatpages import FlatPages
from flask_frozen import Freezer
from werkzeug.utils import secure_filename

from image_processing import process_image

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FLATPAGES_ROOT = 'content'
POST_DIR = 'posts'
UPLOAD_FOLDER = 'static\img\IO_img'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

flatpages = FlatPages(app)
freezer = Freezer(app)
app.config.from_object(__name__)

with open('settings.txt', encoding='utf8') as config:
    data = config.read()
    settings = json.loads(data)


# home page of the site
@app.route('/', methods=['GET'])
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

    output_file, percentage_list, gender = process_image(UPLOAD_FOLDER, filename)
    filename = 'http://127.0.0.1:8000/uploads/' + output_file

    return render_template('loading_form.html', filename=filename, percentage=percentage_list, gender=gender,
                           **settings)


# utility method for loading image in html
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.errorhandler(404)
def page_not_found(e):
    # handdling all Errors
    return render_template("500_generic.html", e=e, **settings), 404

@app.errorhandler(Exception)
def handle_exception(e):
    # handdling all Errors
    return render_template("500_generic.html", e="Проблемы с загрузкой графического файла", **settings), 500

# server launch
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(host='127.0.0.1', port=8000, debug=True)
