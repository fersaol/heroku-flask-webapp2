from flask import Flask, session, url_for, render_template, redirect
from co2_form import Input_Form
import sys
from pathlib import Path
import os
import datetime
sys.path.append(Path.cwd()/"model")
from model import modelo_final_concampos as mymodel

root_os = os.getcwd()

# creamos la app de Flask
app = Flask(__name__,template_folder="templates")
app.config['SECRET_KEY'] = 'mysecretkey'
app.config["DEBUG"] = True

@app.route('/', methods=['GET','POST'])
def index():
    form = Input_Form()

    if form.validate_on_submit():
        session['Country'] = form.Country.data
        session['Year'] = form.Year.data
        session['GDP'] = form.GDP.data
        session['Population'] = form.Population.data
        session['Energy_production'] = form.Energy_production.data
        session['Energy_consumption'] = form.Energy_consumption.data
        session['CO2_emission'] = form.CO2_emission.data
        session['Energy_code'] = form.energy_type.data

        return redirect(url_for('prediction'))
    return render_template("home.html", form=form)

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    day,month,year = session['Year'].split(" ")[1:4]
    date = f"{year}-{month}-{day}"
    results = mymodel.Final_Model(
                            session['Country'],
                            date,
                            float(session['GDP']),
                            float(session['Population']),
                            float(session['Energy_production']),
                            float(session['Energy_consumption']),
                            float(session['CO2_emission']),
                            float(session['Energy_code'])).run_whole_model()

    return render_template("prediction.html", results=results)

# Ejecutamos la aplicación app.run()
if __name__ == '__main__':
    # LOCAL
    # app.run(host='0.0.0.0', port=8080)

    # REMOTO
    app.run()