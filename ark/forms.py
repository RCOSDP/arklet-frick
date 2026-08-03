import re

from django import forms
from django.core.exceptions import ValidationError

from ark.utils import parse_ark

# A shoulder is a sub-namespace of the NAAN. By convention it runs from the end
# of the NAAN up to and including its first digit, which is how a reader finds
# where the shoulder stops and the blade begins without a visual separator. It
# therefore starts with one or more lowercase letters, avoids vowels and the
# letter 'l' so it cannot spell words, and ends with a single digit.
SHOULDER_PATTERN = re.compile(r"^/[bcdfghjkmnpqrstvwxz]+[0-9]$")


def validate_shoulder(shoulder: str):
    """Validate a shoulder against the first-digit convention.

    A slash after the shoulder is specifically warned against: it implies both
    that the part before it names a real object and that the full ARK is
    contained within that object, neither of which is true.
    """
    if not shoulder.startswith("/"):
        raise ValidationError("Shoulders must start with a forward slash")
    if not SHOULDER_PATTERN.match(shoulder):
        raise ValidationError(
            "A shoulder must be a single segment of lowercase betanumeric "
            "consonants ending in one digit, e.g. '/x5'. It may not contain a "
            "further '/' or '.', which would falsely imply containment."
        )


def validate_ark(ark: str):
    try:
        parse_ark(ark)
    except ValueError as e:
        raise ValidationError(f"Invalid ARK: {e}")


class MintArkForm(forms.Form):
    naan = forms.IntegerField()
    shoulder = forms.CharField(validators=[validate_shoulder])
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
