import re

from django import forms
from django.core.exceptions import ValidationError

from ark.utils import parse_ark

# Alphanumeric segments joined by single '.', '-' or '/' separators, with an
# optional trailing separator. Leading separators and repeated separators are
# rejected so that the resulting ARK stays parseable by parse_ark().
NAME_PREFIX_PATTERN = re.compile(r"^(?:[0-9A-Za-z]+[./-]?)+$")
NAME_PREFIX_MAX_LENGTH = 40


def validate_shoulder(shoulder: str):
    if not shoulder.startswith("/"):
        raise ValidationError("Shoulders must start with a forward slash")


def validate_name_prefix(name_prefix: str):
    """Validate a caller-supplied prefix for the assigned name.

    Note that ARK normalization treats hyphens as insignificant while arklet
    resolves on an exact string match, so hyphens in a prefix are best avoided.
    """
    if len(name_prefix) > NAME_PREFIX_MAX_LENGTH:
        raise ValidationError(
            f"Name prefixes are limited to {NAME_PREFIX_MAX_LENGTH} characters"
        )
    if not NAME_PREFIX_PATTERN.match(name_prefix):
        raise ValidationError(
            "Name prefixes must be alphanumeric segments separated by single "
            "'.', '-' or '/' characters and may not start with a separator"
        )


def validate_ark(ark: str):
    try:
        parse_ark(ark)
    except ValueError as e:
        raise ValidationError(f"Invalid ARK: {e}")


class MintArkForm(forms.Form):
    naan = forms.IntegerField()
    shoulder = forms.CharField(validators=[validate_shoulder])
    name_prefix = forms.CharField(required=False, validators=[validate_name_prefix])
    url = forms.URLField(required=False)
    metadata = forms.CharField(required=False)
    title = forms.CharField(required=False)
    type = forms.CharField(required=False)
    commitment = forms.CharField(required=False)
    identifier = forms.CharField(required=False)
    format = forms.CharField(required=False)
    relation = forms.CharField(required=False)
    source = forms.URLField(required=False)


class UpdateArkForm(forms.Form):
    ark = forms.CharField(validators=[validate_ark])
    url = forms.URLField(required=False)
    metadata = forms.CharField(required=False)
    title = forms.CharField(required=False)
    type = forms.CharField(required=False)
    commitment = forms.CharField(required=False)
    identifier = forms.CharField(required=False)
    format = forms.CharField(required=False)
    relation = forms.CharField(required=False)
    source = forms.URLField(required=False)

    def clean(self):
        cleaned_data = super().clean()

        # Remove fields that are not provided in the request
        for field_name in self.fields:
            if self.data.get(field_name) is None:
                cleaned_data.pop(field_name, None)

        return cleaned_data
