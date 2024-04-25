import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__,static_url_path='/static')
app.config["MONGO_URI"] = "mongodb+srv://mohammedfauzan44:3HkTglbR0RVoYDaU@cluster0.cms2ddb.mongodb.net/breast_cancer?retryWrites=true&w=majority"
app.secret_key = "bae603dd08975354ea635bc400914476"
mongo = PyMongo(app)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

#### Routes #######

@app.route('/')
def home():
    if 'loggedin' in session:
        return redirect(url_for('initial'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashpass = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
            users.insert_one({'username': request.form['username'], 'password': hashpass})
            flash('You are successfully registered! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'username': request.form['username']})

        if login_user:
            # Check if the password matches
            if check_password_hash(login_user['password'], request.form['password']):
                # Successful login
                # Here you can set up the user session
                session['username'] = request.form['username']
                return redirect(url_for('initial'))
            else:
                # Invalid password
                flash('Invalid username/password combination.', 'danger')
        else:
            # User not found
            flash('Username does not exist.', 'danger')

    return render_template('login.html')

@app.route('/initial')
def initial():
    # Ensure user is logged in
    if 'username' in session:
        return render_template('initial.html')
    else:
        # Redirect to login if not logged in
        return redirect(url_for('login'))


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/save_patient', methods=['POST'])
def save_patient():
    if 'username' not in session:
        flash('Please log in to proceed.', 'danger')
        return redirect(url_for('login'))

    patient_id = request.form['patientId']
    patient_name = request.form['patientName']
    
    patients = mongo.db.patients
    patient_record = {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'created_by': session['username']
    }
    patients.insert_one(patient_record)
    
    # Save patient_id in session to use in the prediction route
    session['patient_id'] = patient_id
    
    flash('Patient details saved successfully.', 'success')
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the patient_id is in session
    if 'patient_id' not in session:
        flash('No patient ID found. Please start the process again.', 'danger')
        return redirect(url_for('initial'))

    patient_id = session['patient_id']
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3)

    # Update the corresponding patient record with prediction and probability
    mongo.db.patients.update_one(
        {'patient_id': patient_id},
        {'$set': {
            'prediction': int(output),
            'probability': y_prob,
            'features': features,
            'final_features': final_features.tolist()
        }}
    )

    session.pop('patient_id', None)  # Clear the patient_id from session after updating

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE {}'.format(y_prob))
    else:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE {}'.format(y_prob))

        
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



@app.route('/view_predictions')
def view_predictions():
    if 'username' not in session:
        flash('Please log in to view predictions.', 'danger')
        return redirect(url_for('login'))

    # Fetch unique patient records from the database along with their last prediction
    patients_aggregate = mongo.db.patients.aggregate([
        {
            '$group': {
                '_id': '$patient_id',
                'patient_name': {'$last': '$patient_name'},
                'prediction': {'$last': '$prediction'},
                'probability': {'$last': '$probability'}
            }
        },
        {
            '$project': {
                'patient_id': '$_id',
                'patient_name': 1,
                'prediction': 1,
                'probability': 1,
                '_id': 0
            }
        }
    ])

    # Convert the aggregate cursor to a list
    patients = list(patients_aggregate)

    # Render the predictions template, passing the unique patient records
    return render_template('predictions.html', patients=patients)





if __name__ == "__main__":
    app.run(debug=True)