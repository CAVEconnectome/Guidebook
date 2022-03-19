from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, SelectField, FloatField
from wtforms.validators import Optional, ValidationError


class L2SkeletonizeForm(FlaskForm):
    root_id = StringField("Root ID", default="", validators=[])
    root_location = StringField(
        "Root location (optional)", default="", validators=[Optional()]
    )
    root_is_soma = BooleanField("Root location is soma", default=True)
    root_id_from_root_loc = BooleanField("Guess ID from point", default=True)
    split_location = StringField(
        "Restriction point (optional)", default="", validators=[Optional()]
    )
    split_option = SelectField(
        "Direction of restriction",
        choices=[
            ("upstream", "Inward from point"),
            ("downstream", "Outward from point"),
        ],
        default="downstream",
    )
    submit = SubmitField("Generate Neuroglancer Link")
    segmentation_fallback = BooleanField(
        "Use segmentation as fallback (slower but more accurate)", default=False
    )

    def validate_root_id(self, field):
        if len(field.data) == 0:
            if self.root_id_from_root_loc.data:
                if len(self.root_location.data) == 0:
                    raise ValidationError(
                        "Must specify a root location if no root id is provided"
                    )
                else:
                    return True
            else:
                raise ValidationError(
                    "Must either set a root id or guess from root location"
                )
        else:
            return True


class Lvl2PointForm(L2SkeletonizeForm):
    point_option = SelectField(
        "Points",
        choices=[
            ("both", "Branch and End Points"),
            ("bp", "Branch Points"),
            ("ep", "End Points"),
        ],
    )


class Lvl2PathForm(L2SkeletonizeForm):
    num_paths = SelectField(
        "Paths to Sample",
        choices=[
            ("all", "All"),
            ("five", "5"),
            ("ten", "10"),
            ("fifteen", "15"),
        ],
        default="all",
    )
    target_dist = FloatField(
        "Sample Path Length (mm, optional)", default=None, validators=[Optional()]
    )
    exclude_short = BooleanField("Exclude Short Paths", default=True)