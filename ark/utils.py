from typing import Tuple

import secrets

BETANUMERIC = "0123456789bcdfghjkmnpqrstvwxz"


def noid_check_digit(noid: str) -> str:
    """Calculate the check digit for an ARK.

    See: https://metacpan.org/dist/Noid/view/noid#NOID-CHECK-DIGIT-ALGORITHM
    """
    total = 0
    for pos, char in enumerate(noid, start=1):
        score = BETANUMERIC.find(char)
        if score > 0:
            total += pos * score
    remainder = total % 29  # 29 == len(BETANUMERIC)
    return BETANUMERIC[remainder]  # IndexError may be long ARK


def generate_noid(length: int) -> str:
    return "".join(secrets.choice(BETANUMERIC) for _ in range(length))


def parse_ark(ark: str) -> Tuple[str, int, str]:
    parts = ark.split("ark:")
    if len(parts) != 2:
        raise ValueError("Not a valid ARK")
    nma, ark = parts
    ark = ark.lstrip("/")
    parts = ark.split("/")
    if len(parts) < 2:
        raise ValueError("Not a valid ARK")
    naan = parts[0]
    identifier = '/'.join(parts[1:])
    try:
        naan_int = int(naan)
    except ValueError:
        raise ValueError("ARK NAAN must be an integer")

    return nma, naan_int, identifier

def parse_ark_lookup(ark: str) -> str:

    _, naan_int, identifier = parse_ark(ark)
    return f"{naan_int}/{identifier}"


# ARK reserves '/' to mean containment (a component of the object) and '.' to
# mean a variant (another form of the same object). Both open the qualifier
# region that follows a base name, so suffix passthrough scans back through
# either one.
QUALIFIER_SEPARATORS = "/."


def gen_prefixes(ark: str):
    """Yield the ancestors of an ark, longest first.

    An ancestor ends immediately before a qualifier separator, matching the
    resolution rule that a resolver scans back from the end of the supplied
    string and stops at the first ancestor it has stored.
    """
    for i in range(len(ark) - 1, 0, -1):
        if ark[i] in QUALIFIER_SEPARATORS:
            yield ark[:i]


def strip_hyphens(text: str) -> str:
    """Remove hyphens, which are insignificant in an ARK.

    A hyphen may be there for readability or may have crept in when the ARK was
    wrapped across lines, so it must be ignored in lexical comparisons.
    """
    return text.replace("-", "")


def split_after_normalized(text: str, length: int) -> Tuple[str, str]:
    """Split text just after its first `length` significant (non-hyphen) characters.

    The head is compared against stored arks, which is why it is measured
    without hyphens. The tail is returned exactly as supplied, because it is a
    qualifier addressing a resource whose path this resolver does not assign
    and whose hyphens therefore matter.
    """
    seen = 0
    for i, char in enumerate(text):
        if char == "-":
            continue
        if seen == length:
            return text[:i], text[i:]
        seen += 1
    return text, ""
