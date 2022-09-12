from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,DateField,DecimalField,IntegerField

class Input_Form(FlaskForm):
    Country = StringField('Country Name')
    Year = DateField('yyyy-mm-dd')
    GDP = DecimalField('GDP',rounding=2)
    Population = DecimalField('Population',rounding=2)
    Energy_production = DecimalField('Energy Production',rounding=2)
    Energy_consumption = DecimalField('Energy Consumption',rounding=2)
    CO2_emission = DecimalField('Co2 Emitted',rounding=2)
    energy_type = IntegerField("0:renewab.,1:nuclear,2:gas,3:petrol,4:coal")

    submit = SubmitField("Predict")

                