from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, SelectField, RadioField
from wtforms.validators import DataRequired, Optional


class SkeletonizeForm(FlaskForm):
    root_id = StringField('Root ID', validators=[DataRequired()])
    root_location = StringField('Root location')
    root_is_soma = BooleanField('Soma')
    submit = SubmitField('Generate Neuroglancer Link')


class Lvl2SkeletonizeForm(FlaskForm):
    root_id = StringField('Root ID', validators=[DataRequired()])
    root_location = StringField(
        'Root location (optional)', default="", validators=[Optional()])
    point_option = SelectField(
        'Points', choices=[('both', 'Branch and End Points'), ('bp', 'Branch Points'), ('ep', 'End Points')], )
    submit = SubmitField('Generate Neuroglancer Link')
