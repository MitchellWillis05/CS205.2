from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import model as mdl
import strength_exercises as se

app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/begin", methods=['GET', 'POST'])
def begin():
    if request.method == "GET":
        return render_template("begin.html")
    if request.method == "POST":
        data = request.get_json()

        try:
            age = int(data.get('age'))
        except ValueError:
            return jsonify({'message': 'Invalid age'}), 400

        gender = data.get('gender')

        print(f"Received age: {age}, gender: {gender}")
        if gender is None:
            return jsonify({'message': 'Please select your gender'}), 400
        if age == 0:
            return jsonify({'message': 'Please select your age'}), 400

        session['age'] = age
        if gender == 'male':
            gender = 1
        else:
            gender = 0
        session['gender'] = gender
        return jsonify({'message': 'Data received successfully'}), 200


@app.route("/continue", methods=['GET', 'POST'])
def cont():
    if request.method == "GET":
        if session['age'] is not None:
            print(session['age'])
            return render_template("continue.html")
        else:
            return render_template("begin.html")
    if request.method == "POST":
        data = request.get_json()

        try:
            weight = int(data.get('weight'))
        except ValueError:
            return jsonify({'message': 'Invalid weight'}), 400
        try:
            height = int(data.get('height'))
        except ValueError:
            return jsonify({'message': 'Invalid height'}), 400

        print(f"Received weight: {weight}, height: {height}")
        if weight == 0:
            return jsonify({'message': 'Please select your age'}), 400
        if height == 0:
            return jsonify({'message': 'Please select your age'}), 400

        session['weight'] = weight
        session['height'] = height
        return jsonify({'message': 'Data received successfully'}), 200


@app.route("/bodyfat", methods=['GET', 'POST'])
def bodyfat():
    if request.method == "GET":
        if session['age'] & session['weight'] is not None:
            return render_template("bodyfat.html")
        else:
            return render_template("begin.html")
    if request.method == "POST":
        data = request.get_json()

        try:
            bodyfat = int(data.get('bodyfat'))
        except ValueError:
            return jsonify({'message': 'Invalid bodyfat'}), 400

        print(f"Received bodyfat: {bodyfat}")
        if bodyfat == 0:
            return jsonify({'message': 'Please enter bodyfat'}), 400

        session['bodyfat'] = bodyfat
        result = mdl.predict_workout(session['age'], session['gender'], session['weight'],
                                     session['height'], session['bodyfat'])
        session["result"] = result
        return jsonify({'message': 'Data received successfully'}), 200


@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == "GET":
        if session['result'] is not None:
            return render_template("result.html", session=session["result"])
        else:
            return render_template("begin.html")
    if request.method == "POST":

        return jsonify({'message': 'Method not allowed'}), 400


@app.route("/exercises", methods=['GET', 'POST'])
def exercises():
    if request.method == "GET":
        if session['result'] is not None:

            return render_template("exercises.html", session=session["result"])
        else:
            return render_template("begin.html")
    if request.method == "POST":
        data = request.get_json()

        equipment = data.get('equipment')
        bodypart = data.get('bodypart')
        proper = session['result'][0].upper() + session['result'][1:] if session['result'] else session['result']
        print(equipment, bodypart, proper)
        exercise = se.recommend_exercises(proper, bodypart, equipment)
        print(exercise)
        session["exercise"] = exercise

        return jsonify({'message': 'Data received successfully'}), 200


@app.route("/exercise-result", methods=['GET', 'POST'])
def exerciseresult():
    if request.method == "GET":
        if session['exercise'] is not None:
            return render_template("exerciseresult.html", session=session["exercise"])
        else:
            return render_template("begin.html")
    if request.method == "POST":

        return jsonify({'message': 'Method not allowed'}), 400




if __name__ == '__main__':
    app.run(debug=True)
