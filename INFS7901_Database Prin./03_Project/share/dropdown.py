from flask import Flask, render_template, request
from database import Database

app = Flask(__name__)
app.debug = True



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form
        name = userDetails['name']
        email = userDetails['email']
        requesttype = userDetails.get('RequestType')
        con = Database.connect()
        cursor = con.cursor()
        cursor.execute("""INSERT INTO users (name, email, requesttype) VALUES (%s, %s, %s)""", (name, email, requesttype))
        con.commit()
        return 'Save is Successful'
    return render_template('index.html')

if __name__ == "__main__":
    app.run()