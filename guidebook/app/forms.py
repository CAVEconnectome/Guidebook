from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired


class SkeletonizeForm(FlaskForm):
    root_id = StringField('Root ID', validators=[DataRequired()])
    root_location = StringField('Root location')
    root_is_soma = BooleanField('Soma')
    submit = SubmitField('Generate Neuroglancer Link')


class Lvl2SkeletonizeForm(FlaskForm):
    root_id = StringField('Root ID', validators=[DataRequired()])
    # root_location = StringField('Root location')
    # root_is_soma = BooleanField('Soma')
    submit = SubmitField('Generate Neuroglancer Link')
