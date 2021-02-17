from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, SelectField, RadioField
from wtforms.validators import DataRequired, Optional


class Lvl2SkeletonizeForm(FlaskForm):
    root_id = StringField('Root ID', validators=[DataRequired()])
    root_location = StringField(
        'Root location (optional)', default="", validators=[Optional()])
    root_is_soma = BooleanField('Root location is soma', default=True)
    point_option = SelectField(
        'Points', choices=[('both', 'Branch and End Points'),
                           ('bp', 'Branch Points'),
                           ('ep', 'End Points')], )
    segmentation_fallback = BooleanField('Use segmentation as fallback (slower but more accurate)',
                                         default=False)
    submit = SubmitField('Generate Neuroglancer Link')