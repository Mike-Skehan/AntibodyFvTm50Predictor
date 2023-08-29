from flask import Flask, request, jsonify, render_template
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv
import numpy as np
import sys
sys.path.append('..')
from pipelines import svm_pipe


load_dotenv('../.env')

# Initialize the Flask app
app = Flask(__name__, static_folder='./static')


# Email configurations
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')

mail = Mail(app)


def predict_melting_temperature(h, l):
    #output = len(h) + len(l)
    output = svm_pipe.predict([h, l])
    return output


@app.route('/')
def home():
    return render_template('index.html')

aa3_aa1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
           'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
           'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
           'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the amino acid sequences from the form data
    heavy_chain = request.form['heavy_chain']
    light_chain = request.form['light_chain']

    # Validate the amino acid sequences
    if not all(aa_char in aa for aa_char in heavy_chain):
        return jsonify({'error': 'Invalid character in heavy chain'})
    if not all(aa_char in aa for aa_char in light_chain):
        return jsonify({'error': 'Invalid character in light chain'})

    # Call the pipeline to predict the melting temperature
    melting_temperature = (predict_melting_temperature(heavy_chain, light_chain))

    # Return the result as JSON
    return jsonify({'melting_temperature': melting_temperature[0]})


@app.route('/Contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Send email
        msg = Message('New message from your website!', sender=os.getenv('EMAIL_USERNAME'),
                      recipients=[os.getenv('EMAIL_USERNAME')])
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        mail.send(msg)

        return render_template('contact.html', success=True)
    else:
        return render_template('contact.html', success=False)


@app.route('/About')
def about():
    return render_template('about.html')