
import os
from flask import Flask, flash, render_template, redirect, url_for, request, session
from database import Database

app = Flask(__name__)
app.secret_key = os.urandom(12)
db = Database()

@app.route('/')
def index():
    data = db.read(None)
    return render_template('index.html', data = data)



####EMPLOYEE####
################
@app.route('/employee-contact')
def employee():
    data = db.read_employee(None)
    return render_template('employee-contact.html', data = data)

@app.route('/employee-grade')
def employee_grade():
    data = db.employee_grade(None)
    return render_template('employee-grade.html', data = data)


@app.route('/employee-form')
def employeeform():
    return render_template('employee-form.html')


@app.route('/addemployee', methods = ['POST', 'GET'])
def addemployee():
    if request.method == 'POST' and request.form['save']:
        if db.insert(request.form):
            flash("A new employee has been added")
        else:
            flash("A new employee can not be added")
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/employee-update/<int:id>/')
def employee_update(id):
    data = db.read(id)
    if len(data) == 0:
        return redirect(url_for('index'))
    else:
        session['update'] = id
        return render_template('employee-update.html', data = data)


@app.route('/updateemp', methods = ['POST'])
def updatephone():
    if request.method == 'POST' and request.form['update']:
        if db.update(session['update'], request.form):
            flash('A employee has been updated')
        else:
            flash('A employee can not be updated')
        session.pop('update', None)
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/updateemployee', methods = ['POST'])
def updateemployee():
    if request.method == 'POST' and request.form['employee-update']:
        if db.employee_update(session['employee-update'], request.form):
            flash('data has been updated')
        else:
            flash('data can not be updated')
        session.pop('employee-update', None)
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/employee-delete/<int:id>/')
def deleteemployee(id):
    data = db.read_employee(id)
    if len(data) == 0:
        return redirect(url_for('index'))
    else:
        session['delete'] = id
        return render_template('employee-delete.html', data = data)


@app.route('/deleteemp', methods = ['POST'])
def deletephone():
    if request.method == 'POST' and request.form['delete']:
        if db.delete(session['delete']):
            flash('A phone number has been deleted')
        else:
            flash('A phone number can not be deleted')
        session.pop('delete', None)
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))




####PRODUCT####
###############
@app.route('/product-list')
def product_list():
    data = db.read_ploduct(None)
    return render_template('product-list.html', data = data)


@app.route('/product-form')
def productform():
    return render_template('product-form.html')





####PROJECT####
###############
@app.route('/project-list')
def project():
    data = db.read_project(None)
    return render_template('project-list.html', data = data)


@app.route('/project-team-members')
def project_teams():
    data = db.project_teams(None)
    return render_template('project-team-members.html', data = data)




@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)






