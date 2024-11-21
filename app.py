from flask import Flask, render_template, request, redirect, url_for, jsonify, session

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
        session['gender'] = gender
        return jsonify({'message': 'Data received successfully'}), 200




if __name__ == '__main__':
    app.run(debug=True)