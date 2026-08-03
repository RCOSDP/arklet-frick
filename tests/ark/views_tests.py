"""Tests for ark/views.py, comprising the main endpoints for arklet."""

import uuid
from dataclasses import asdict, dataclass
from itertools import chain, count
from unittest.mock import patch

import pytest
from django.core.exceptions import ValidationError

from ark.forms import validate_shoulder
from ark.models import Ark, Key, Naan, Shoulder
from ark.utils import noid_check_digit, parse_ark
from arklet.settings import env


@dataclass
class MintArkArgs:
    """Django test client named arguments to test mint_ark.

    Example use: client.post(**asdict(mint_ark_args))
    """

    path: str
    data: dict
    content_type: str
    HTTP_AUTHORIZATION: str  # pylint: disable=invalid-name


@pytest.fixture
def naan(db):
    """Create the initial NAAN used for most tests."""
    return Naan.objects.create(
        naan=1, name="Archive", description="A NAAN", url="https://example.com"
    )


@pytest.fixture
def shoulder(db, naan):
    """Create an initial shoulder used for most tests."""
    return Shoulder.objects.create(
        shoulder="/t2", naan=naan, name="Test", description="A Shoulder"
    )


@pytest.fixture
def auth(db, naan):
    """Create an access key for the initial naan.

    Keys are stored hashed, so the bearer token is the raw api key returned by
    create_for_naan, not the persisted Key.key value.
    """
    _, api_key = Key.create_for_naan(naan.naan)
    return f"Bearer {api_key}"


@pytest.fixture
def ark(db, naan, shoulder):
    """Create an ARK for tests."""
    return Ark.objects.create(
        ark=f"{naan.naan}{shoulder.shoulder}12346",
        naan=naan,
        shoulder=shoulder,
        assigned_name="12346",
    )


@pytest.fixture
def mint_ark_args(naan, shoulder, auth) -> MintArkArgs:
    """Create the happy path arguments for mint_ark in Django test client."""
    return MintArkArgs(
        path="/mint",
        data={
            "naan": naan.naan,
            "shoulder": shoulder.shoulder,
        },
        content_type="application/json",
        HTTP_AUTHORIZATION=auth,
    )


class TestMintArk:
    """Test the arklet mint_ark endpoint.

    mint_ark is responsible for the creation of new ARKs.
    """

    @staticmethod
    def _validate_success(test_args, res) -> None:
        """mint_ark returns a 200 and a json payload with a valid ARK on success."""
        minted_ark = res.json()["ark"]
        _, minted_naan, minted_assigned_name = parse_ark(minted_ark)
        expected_naan = test_args.data["naan"]
        expected_assigned_name = test_args.data["shoulder"].lstrip("/")
        assert res.status_code == 200
        assert minted_naan == expected_naan
        assert minted_assigned_name.startswith(expected_assigned_name)

    @pytest.mark.django_db
    def test_happy_path(self, client, mint_ark_args) -> None:
        """mint_ark succeeds on the happy path."""
        res = client.post(**asdict(mint_ark_args))
        self._validate_success(mint_ark_args, res)

    @pytest.mark.django_db
    def test_post_only(self, client, mint_ark_args) -> None:
        """mint_ark only accepts POST requests."""
        # When using client.put instead of client.post
        res = client.put(**asdict(mint_ark_args))
        # Then we get a 405
        assert res.status_code == 405

    def test_only_accepts_json(self, client, mint_ark_args) -> None:
        """mint_ark only accepts JSON payloads."""
        args = asdict(mint_ark_args)
        # When a request is sent form encoded instead of application/json
        del args["content_type"]
        res = client.post(**args)
        # Then we get a 400 Bad Request
        assert res.status_code == 400

    def test_invalid_form_is_bad_request(self, client, mint_ark_args) -> None:
        """mint_ark doesn't crash when sent JSON with the wrong structure."""
        mint_ark_args.data = {"a": "b"}
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 400

    def test_http_authorization_header_required(self, client, mint_ark_args) -> None:
        """mint_ark requires an HTTP_AUTHORIZATION header."""
        # When we send a request with no HTTP_AUTHORIZATION header
        args = asdict(mint_ark_args)
        del args["HTTP_AUTHORIZATION"]
        res = client.post(**args)
        # Then we get a 403 Forbidden
        assert res.status_code == 403

    def test_verify_key_has_naan(self, client, mint_ark_args) -> None:
        """mint_ark requires present auth header to link to a NAAN."""
        # When we create an auth header value with a random UUID4
        mint_ark_args.HTTP_AUTHORIZATION = f"Bearer {uuid.uuid4()}"
        res = client.post(**asdict(mint_ark_args))
        # Then there will be no NAAN in the database with a key with that UUID4 value.
        assert res.status_code == 403

    def test_verify_key_is_valid(self, client, mint_ark_args) -> None:
        """mint_ark rejects a key that isn't a uuid4.

        Keys are compared as hashed passwords, so a malformed key is simply not
        a match and is refused without disclosing why.
        """
        # When the authorization header value isn't a UUID4
        mint_ark_args.HTTP_AUTHORIZATION = "Bearer not-a-uuid4"
        res = client.post(**asdict(mint_ark_args))
        # Then we get a 403 Forbidden
        assert res.status_code == 403

    def test_authorized_naan_matches_post_naan(self, client, mint_ark_args) -> None:
        """mint_ark NAAN in auth header matches NAAN in POST body."""
        # When the POST body naan field doesn't match the auth header NAAN
        mint_ark_args.data["naan"] += 1
        res = client.post(**asdict(mint_ark_args))
        # Then we get a 403 Forbidden
        assert res.status_code == 403

    @pytest.mark.django_db(transaction=True)
    @patch("ark.models.generate_noid")
    def test_fails_after_too_many_collisions(
        self, mock_noid_gen, caplog, client, mint_ark_args, ark
    ) -> None:
        """mint_ark returns an error after too many collisions.

        We need to set `pytest.mark.django_db(transaction=True)` because this test
        deliberately causes IntegrityErrors. When transaction=False all tests occur
        within a transaction. That prevents the test from using transaction features
        within the test. `transaction=True` is equivalent to Django
        TransactionTestCase. `transaction=False` is equivalent to Django TestCase.

        We patch ark.models.generate_noid (even though generate_noid is originally
        defined in ark.utils) because Ark.create imports it directly into
        ark.models.
        """
        # pylint: disable=too-many-arguments
        # When mint_ark keeps creating NOIDs that collide with an existing ARK
        existing_noid = ark.assigned_name[:-1]
        mock_noid_gen.return_value = existing_noid
        res = client.post(**asdict(mint_ark_args))
        # Then we log the error
        msg = "Gave up creating ark after"
        assert any(record for record in caplog.records if record.msg.startswith(msg))
        # Then we get a 500 Internal Server Error
        assert res.status_code == 500

    @pytest.mark.django_db(transaction=True)
    @patch("ark.models.generate_noid")
    def test_succeeds_on_single_collision(
        self, mock_noid_gen, caplog, client, mint_ark_args, ark
    ) -> None:
        """mint_ark succeeds with collisions, but fewer than the max collisions.

        mint_ark will also log a warning when collisions occur.

        See notes for test_fails_after_too_many_collisions.
        """
        # pylint: disable=too-many-arguments

        # mock generate_noid to return a conflicting NOID on first call
        # and non-conflicting NOIDs on subsequent calls
        existing_noid = ark.assigned_name[:-1]
        non_colliding_noid_gen = (str(i) for i in count(100_000_000))
        return_values = chain([existing_noid], non_colliding_noid_gen)
        mock_noid_gen.side_effect = lambda noid_length: next(return_values)

        # When mint_ark generates a single collision
        res = client.post(**asdict(mint_ark_args))
        # Then arklet logs a warning about the collision, but otherwise succeeds
        msg = "Ark created after %d collision(s)"
        assert any(record for record in caplog.records if record.msg == msg)
        self._validate_success(mint_ark_args, res)


@pytest.fixture
def nested_shoulder(db, naan):
    """Create a second shoulder, used to build arks with qualifiers.

    Containment is expressed in the qualifier region after a base name, never
    inside the shoulder, so this is an ordinary single-segment shoulder.
    """
    return Shoulder.objects.create(
        shoulder="/c5", naan=naan, name="Collections", description="A Shoulder"
    )


@pytest.fixture
def other_naan(db):
    """Create a second NAAN, used to test cross-NAAN isolation."""
    return Naan.objects.create(
        naan=2, name="Other", description="Another NAAN", url="https://other.example.com"
    )


class TestMintedName:
    """Test the shape of a minted name: shoulder, then NOID and check digit."""

    @staticmethod
    def _blade(minted_ark, naan, shoulder):
        """Return the part of the name that follows the shoulder.

        parse_ark reports everything after the NAAN, so the shoulder has to be
        taken off before the generated name can be inspected.
        """
        _, minted_naan, name = parse_ark(minted_ark)
        assert minted_naan == naan.naan
        expected_shoulder = shoulder.shoulder.lstrip("/")
        assert name.startswith(expected_shoulder)
        return name[len(expected_shoulder):]

    @classmethod
    def _assert_valid_name(cls, minted_ark, naan, shoulder):
        """The name must be exactly a NOID plus a verifying check digit."""
        blade = cls._blade(minted_ark, naan, shoulder)
        assert len(blade) == env("ARKLET_NOID_LENGTH") + 1  # noid + check digit
        noid, check_digit = blade[:-1], blade[-1]
        base = f"{naan.naan}{shoulder.shoulder}{noid}"
        assert noid_check_digit(base) == check_digit

    @pytest.mark.django_db
    def test_name_is_a_noid_and_its_check_digit(
        self, client, mint_ark_args, naan, shoulder
    ) -> None:
        """Nothing precedes the generated name, so it carries no meaning."""
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 200
        self._assert_valid_name(res.json()["ark"], naan, shoulder)

    @pytest.mark.django_db
    def test_second_shoulder_divides_the_namespace(
        self, client, mint_ark_args, naan, nested_shoulder
    ) -> None:
        """A namespace is divided by registering another shoulder."""
        mint_ark_args.data["shoulder"] = nested_shoulder.shoulder
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 200
        minted_ark = res.json()["ark"]
        # The blade follows the shoulder directly, with no separator between
        assert minted_ark.startswith("ark:/1/c5")
        self._assert_valid_name(minted_ark, naan, nested_shoulder)

    @pytest.mark.django_db
    def test_a_caller_cannot_prepend_to_the_name(
        self, client, mint_ark_args, naan, shoulder
    ) -> None:
        """A caller has no way to put its own characters in front of the name.

        Anything accepted here would be read as meaning by whoever sees the
        ark, and a namespace is divided by registering a shoulder instead.
        """
        mint_ark_args.data["name_prefix"] = "2026"
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 200
        blade = self._blade(res.json()["ark"], naan, shoulder)
        assert not blade.startswith("2026")
        assert len(blade) == env("ARKLET_NOID_LENGTH") + 1

    @pytest.mark.django_db(transaction=True)
    @patch("ark.models.generate_noid")
    def test_collision_retry_keeps_the_registered_ark(
        self, mock_noid_gen, caplog, client, mint_ark_args, naan, shoulder
    ) -> None:
        """A NOID collision retries rather than retargeting the stored ark."""
        # pylint: disable=too-many-arguments
        colliding_noid = "12345678"
        base = f"{naan.naan}{shoulder.shoulder}{colliding_noid}"
        colliding_ark = Ark.objects.create(
            ark=f"{base}{noid_check_digit(base)}",
            naan=naan,
            shoulder=shoulder,
            assigned_name=f"{colliding_noid}{noid_check_digit(base)}",
            url="https://example.com/original",
        )
        non_colliding = (str(i) for i in count(100_000_000))
        return_values = chain([colliding_noid], non_colliding)
        mock_noid_gen.side_effect = lambda noid_length: next(return_values)

        res = client.post(**asdict(mint_ark_args))
        # Then minting succeeds without touching the colliding ark
        assert res.status_code == 200
        assert Ark.objects.count() == 2
        assert res.json()["ark"] != f"ark:/{colliding_ark.ark}"
        colliding_ark.refresh_from_db()
        assert colliding_ark.url == "https://example.com/original"
        # And the collision is logged as a warning
        assert any(
            record for record in caplog.records
            if record.msg == "Ark created after %d collision(s)"
        )


class TestShoulderIsScopedToNaan:
    """A shoulder registered under one NAAN must not be usable by another."""

    @pytest.mark.django_db
    def test_mint_rejects_shoulder_of_another_naan(
        self, client, mint_ark_args, other_naan
    ) -> None:
        """mint_ark refuses a shoulder that belongs to a different NAAN."""
        foreign = Shoulder.objects.create(
            shoulder="/x9", naan=other_naan, name="Foreign", description="Other NAAN"
        )
        mint_ark_args.data["shoulder"] = foreign.shoulder
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 400
        assert not Ark.objects.exists()

    @pytest.mark.django_db
    def test_bulk_mint_rejects_shoulder_of_another_naan(
        self, client, naan, auth, other_naan
    ) -> None:
        """bulk_mint refuses a shoulder that belongs to a different NAAN."""
        Shoulder.objects.create(
            shoulder="/x9", naan=other_naan, name="Foreign", description="Other NAAN"
        )
        res = client.post(
            path="/bulk_mint",
            data={"naan": naan.naan, "data": [{"shoulder": "/x9"}]},
            content_type="application/json",
            HTTP_AUTHORIZATION=auth,
        )
        assert res.status_code == 400
        assert not Ark.objects.exists()


class TestBulkMintNames:
    """Test that a batch mints names the same way a single request does."""

    @pytest.mark.django_db
    def test_every_record_gets_a_generated_name(
        self, client, naan, shoulder, auth
    ) -> None:
        """No record in a batch can carry its own prefix into the name."""
        res = client.post(
            path="/bulk_mint",
            data={
                "naan": naan.naan,
                "data": [
                    {"shoulder": shoulder.shoulder, "name_prefix": "2025"},
                    {"shoulder": shoulder.shoulder, "name_prefix": "2026"},
                    {"shoulder": shoulder.shoulder},
                ],
            },
            content_type="application/json",
            HTTP_AUTHORIZATION=auth,
        )
        assert res.status_code == 200
        assert res.json()["num_received"] == 3
        names = [a.assigned_name for a in Ark.objects.all()]
        assert len(names) == 3
        expected = env("ARKLET_NOID_LENGTH") + 1
        assert all(len(n) == expected for n in names)
        assert not any(n.startswith("2025") or n.startswith("2026") for n in names)


class TestResolveLongestPrefix:
    """Test suffix passthrough when several ancestors of an ARK are registered."""

    @staticmethod
    def _create_ark(naan, shoulder, assigned_name, url):
        return Ark.objects.create(
            ark=f"{naan.naan}{shoulder.shoulder}{assigned_name}",
            naan=naan,
            shoulder=shoulder,
            assigned_name=assigned_name,
            url=url,
        )

    @pytest.mark.django_db
    def test_version_suffix_uses_nearest_ancestor(
        self, client, naan, nested_shoulder
    ) -> None:
        """A /v2 suffix resolves against the item, not a shorter ancestor ark."""
        # Given a collection-level ARK and an item-level ARK beneath it, so that
        # the requested ark has two registered ancestors
        self._create_ark(naan, nested_shoulder, "col5", "https://example.com/collection")
        item = self._create_ark(
            naan, nested_shoulder, "col5/k3n7q9wb2", "https://example.com/item"
        )
        # When resolving an unregistered version of the item
        res = client.get(f"/ark:/{item.ark}/v2")
        # Then we are redirected to the item, with the suffix passed through
        assert res.status_code == 302
        assert res.url == "https://example.com/item/v2"

    @pytest.mark.django_db
    def test_unregistered_intermediate_level_falls_back(
        self, client, naan, nested_shoulder
    ) -> None:
        """With no item-level ARK, the collection-level ARK still answers."""
        collection = self._create_ark(
            naan, nested_shoulder, "col5", "https://example.com/collection"
        )
        res = client.get(f"/ark:/{collection.ark}/k3n7q9wb2/v2")
        assert res.status_code == 302
        assert res.url == "https://example.com/collection/k3n7q9wb2/v2"

    @pytest.mark.django_db
    def test_exact_match_is_preferred_over_prefix(
        self, client, naan, nested_shoulder
    ) -> None:
        """A registered version ARK resolves to its own url."""
        self._create_ark(
            naan, nested_shoulder, "col5/k3n7q9wb2", "https://example.com/item"
        )
        version = self._create_ark(
            naan, nested_shoulder, "col5/k3n7q9wb2/v2", "https://example.com/item-v2"
        )
        res = client.get(f"/ark:/{version.ark}")
        assert res.status_code == 302
        assert res.url.startswith("https://example.com/item-v2")

    @pytest.mark.django_db
    def test_suffix_on_url_less_ancestor_shows_metadata(
        self, client, naan, nested_shoulder
    ) -> None:
        """A suffix cannot pass through an ark that has no url registered."""
        # Given an ark reserved without a target url
        reserved = self._create_ark(naan, nested_shoulder, "col5", "")
        reserved.title = "Reserved collection"
        reserved.save()
        # When resolving a suffixed form of it
        res = client.get(f"/ark:/{reserved.ark}/v2")
        # Then we describe the ancestor rather than redirecting to a bare suffix
        assert res.status_code == 200
        assert "Reserved collection" in res.content.decode()
        assert reserved.ark in res.content.decode()

    @pytest.mark.django_db
    def test_url_less_exact_match_shows_metadata(
        self, client, naan, nested_shoulder
    ) -> None:
        """An ark with no url resolves to arklet's own metadata page."""
        reserved = self._create_ark(naan, nested_shoulder, "col5", "")
        res = client.get(f"/ark:/{reserved.ark}")
        assert res.status_code == 200
        assert reserved.ark in res.content.decode()


class TestInflectionOnSuffix:
    """Test that ?info and ?json answer for a suffix, not redirect past it.

    arklet is the end of the resolution chain, so a metadata request for an
    unminted sub-resource has to be answered from the nearest registered
    ancestor rather than forwarded on.
    """

    @staticmethod
    def _create_ark(naan, shoulder, assigned_name, url, **fields):
        return Ark.objects.create(
            ark=f"{naan.naan}{shoulder.shoulder}{assigned_name}",
            naan=naan,
            shoulder=shoulder,
            assigned_name=assigned_name,
            url=url,
            **fields,
        )

    @pytest.mark.django_db
    def test_json_inflection_on_suffix_returns_metadata(
        self, client, naan, nested_shoulder
    ) -> None:
        """?json on an unminted sub-resource is answered, not redirected."""
        # Given a registered collection with metadata
        collection = self._create_ark(
            naan,
            nested_shoulder,
            "col5",
            "https://example.com/collection",
            title="Scan collection",
            format="application/x-nexus",
        )
        # When asking for the metadata of a sub-resource that was never minted
        res = client.get(f"/ark:/{collection.ark}/entry/instrument?json")
        # Then the request is answered rather than redirected
        assert res.status_code == 200
        body = res.json()
        # And the record is about the ark that was requested
        assert body["ark"]["value"] == f"{collection.ark}/entry/instrument"
        # And it says whose metadata it is, and which part was not registered
        assert body["inherited_from"]["value"] == collection.ark
        assert body["suffix"]["value"] == "/entry/instrument"
        # And the values are inherited from that ancestor
        assert body["title"]["value"] == "Scan collection"
        assert body["format"]["value"] == "application/x-nexus"
        # And url points at where the requested ark actually resolves
        assert body["url"]["value"] == "https://example.com/collection/entry/instrument"

    @pytest.mark.django_db
    def test_info_inflection_on_suffix_returns_metadata(
        self, client, naan, nested_shoulder
    ) -> None:
        """?info on an unminted sub-resource renders the ancestor's metadata."""
        collection = self._create_ark(
            naan,
            nested_shoulder,
            "col5",
            "https://example.com/collection",
            title="Scan collection",
        )
        res = client.get(f"/ark:/{collection.ark}/entry/instrument?info")
        assert res.status_code == 200
        content = res.content.decode()
        assert f"{collection.ark}/entry/instrument" in content
        assert "Scan collection" in content
        assert "inherited" in content.lower()

    @pytest.mark.django_db
    def test_inflection_uses_nearest_ancestor(
        self, client, naan, nested_shoulder
    ) -> None:
        """Metadata is inherited from the longest registered ancestor."""
        # Given both a collection and an item beneath it
        self._create_ark(
            naan,
            nested_shoulder,
            "col5",
            "https://example.com/collection",
            title="Collection",
        )
        item = self._create_ark(
            naan,
            nested_shoulder,
            "col5/k3n7q9wb2",
            "https://example.com/item",
            title="Item",
        )
        # When asking about a sub-resource of the item
        res = client.get(f"/ark:/{item.ark}/v2?json")
        # Then the item answers, not the collection
        body = res.json()
        assert body["inherited_from"]["value"] == item.ark
        assert body["title"]["value"] == "Item"
        assert body["url"]["value"] == "https://example.com/item/v2"

    @pytest.mark.django_db
    def test_inflection_on_url_less_ancestor_keeps_url_empty(
        self, client, naan, nested_shoulder
    ) -> None:
        """A reserved ancestor has no target, so the suffix resolves nowhere."""
        reserved = self._create_ark(
            naan, nested_shoulder, "col5", "", title="Reserved collection"
        )
        res = client.get(f"/ark:/{reserved.ark}/v2?json")
        assert res.status_code == 200
        body = res.json()
        # The url stays empty rather than becoming the bare suffix "/v2"
        assert body["url"]["value"] == ""
        assert body["inherited_from"]["value"] == reserved.ark
        assert body["title"]["value"] == "Reserved collection"

    @pytest.mark.django_db
    def test_exact_match_json_has_no_inheritance_fields(
        self, client, naan, nested_shoulder
    ) -> None:
        """An exactly matched ark answers for itself, with the shape unchanged."""
        item = self._create_ark(
            naan, nested_shoulder, "col5", "https://example.com/collection"
        )
        body = client.get(f"/ark:/{item.ark}?json").json()
        assert body["ark"]["value"] == item.ark
        assert "inherited_from" not in body
        assert "suffix" not in body

    @pytest.mark.django_db
    def test_json_does_not_leak_values_between_requests(
        self, client, naan, nested_shoulder
    ) -> None:
        """Serializing must not write values into the class level COLUMN_METADATA."""
        # Given two arks with different titles
        first = self._create_ark(
            naan, nested_shoulder, "col5", "https://example.com/a", title="First"
        )
        second = self._create_ark(
            naan, nested_shoulder, "col6", "https://example.com/b", title="Second"
        )
        # When both are serialized in turn
        client.get(f"/ark:/{first.ark}?json")
        body = client.get(f"/ark:/{second.ark}?json").json()
        # Then the second response carries its own values
        assert body["title"]["value"] == "Second"
        # And the shared definition was left clean
        assert "value" not in Ark.COLUMN_METADATA["title"]

    @pytest.mark.django_db
    def test_uninflected_suffix_still_redirects(
        self, client, naan, nested_shoulder
    ) -> None:
        """Passthrough is unchanged when no inflection is requested."""
        collection = self._create_ark(
            naan, nested_shoulder, "col5", "https://example.com/collection"
        )
        res = client.get(f"/ark:/{collection.ark}/entry/instrument")
        assert res.status_code == 302
        assert res.url == "https://example.com/collection/entry/instrument"


class TestShoulderShape:
    """Test that a shoulder stays a single betanumeric segment.

    The shoulder is where a namespace is divided. It runs to the first digit so
    that its end is recognisable without a separator, and adding a '/' after it
    would falsely imply that the prefix names a containing object.
    """

    @pytest.mark.django_db
    @pytest.mark.parametrize("good", ["/x5", "/t2", "/nx1", "/jc2", "/r1"])
    def test_conventional_shoulders_are_accepted(self, good) -> None:
        validate_shoulder(good)

    @pytest.mark.django_db
    @pytest.mark.parametrize(
        "bad",
        [
            "x5",  # no leading slash
            "/jc2/nii/",  # a namespace built from slashes
            "/x5/",  # trailing slash after the shoulder
            "/x5w",  # does not end at a digit, so the blade cannot be found
            "/ae1",  # vowels can spell words
            "/l1",  # 'l' is excluded along with the vowels
            "/x5.v",  # '.' marks a variant, not a namespace
            "/5",  # no letters
        ],
    )
    def test_unconventional_shoulders_are_rejected(self, bad) -> None:
        with pytest.raises(ValidationError):
            validate_shoulder(bad)

    @pytest.mark.django_db
    def test_minting_with_a_nested_shoulder_is_refused(
        self, client, mint_ark_args, naan
    ) -> None:
        """A shoulder holding slashes cannot be used to mint, even if stored."""
        Shoulder.objects.create(
            shoulder="/jc2/nii/", naan=naan, name="Legacy", description="Nested"
        )
        mint_ark_args.data["shoulder"] = "/jc2/nii/"
        res = client.post(**asdict(mint_ark_args))
        assert res.status_code == 400
        assert not Ark.objects.exists()


class TestVariantQualifier:
    """Test passthrough across '.', which marks a variant of the same object."""

    @staticmethod
    def _create_ark(naan, shoulder, assigned_name, url):
        return Ark.objects.create(
            ark=f"{naan.naan}{shoulder.shoulder}{assigned_name}",
            naan=naan,
            shoulder=shoulder,
            assigned_name=assigned_name,
            url=url,
        )

    @pytest.mark.django_db
    def test_variant_falls_back_to_the_base_object(
        self, client, naan, nested_shoulder
    ) -> None:
        """ark:/1/c5run1.mzml resolves through the base object, not past it."""
        base = self._create_ark(
            naan, nested_shoulder, "run1", "https://example.com/run.mzMLb"
        )
        res = client.get(f"/ark:/{base.ark}.mzml")
        assert res.status_code == 302
        assert res.url == "https://example.com/run.mzMLb.mzml"

    @pytest.mark.django_db
    def test_variant_of_a_component_uses_the_component(
        self, client, naan, nested_shoulder
    ) -> None:
        """Scanning stops at the nearest ancestor, across either separator."""
        self._create_ark(naan, nested_shoulder, "run1", "https://example.com/base")
        component = self._create_ark(
            naan, nested_shoulder, "run1/spectra", "https://example.com/spectra"
        )
        res = client.get(f"/ark:/{component.ark}.json")
        assert res.status_code == 302
        assert res.url == "https://example.com/spectra.json"

    @pytest.mark.django_db
    def test_registered_variant_wins_over_passthrough(
        self, client, naan, nested_shoulder
    ) -> None:
        """A variant that was minted resolves to its own target."""
        self._create_ark(naan, nested_shoulder, "run1", "https://example.com/base")
        variant = self._create_ark(
            naan, nested_shoulder, "run1.mzml", "https://example.com/converted"
        )
        res = client.get(f"/ark:/{variant.ark}")
        assert res.status_code == 302
        assert res.url.startswith("https://example.com/converted")

    @pytest.mark.django_db
    def test_variant_inflection_reports_the_base(
        self, client, naan, nested_shoulder
    ) -> None:
        """Metadata for an unminted variant is inherited from its base object."""
        base = self._create_ark(
            naan, nested_shoulder, "run1", "https://example.com/run.mzMLb"
        )
        body = client.get(f"/ark:/{base.ark}.mzml?json").json()
        assert body["inherited_from"]["value"] == base.ark
        assert body["suffix"]["value"] == ".mzml"


class TestHyphensAreInsignificant:
    """Test that hyphens are ignored when an ark is looked up.

    A hyphen may be added for readability or picked up when an ark is wrapped
    across lines, so it must not change which object is found.
    """

    @staticmethod
    def _create_ark(naan, shoulder, assigned_name, url):
        return Ark.objects.create(
            ark=f"{naan.naan}{shoulder.shoulder}{assigned_name}",
            naan=naan,
            shoulder=shoulder,
            assigned_name=assigned_name,
            url=url,
        )

    @pytest.mark.django_db
    def test_hyphenated_ark_resolves(self, client, naan, nested_shoulder) -> None:
        """A hyphen inside the name does not stop the ark from resolving."""
        ark = self._create_ark(
            naan, nested_shoulder, "k3n7q9wb2", "https://example.com/item"
        )
        res = client.get(f"/ark:/{naan.naan}{nested_shoulder.shoulder}k3n7-q9wb2")
        assert res.status_code == 302
        assert res.url.startswith("https://example.com/item")

    @pytest.mark.django_db
    def test_hyphen_in_the_name_does_not_reach_the_target(
        self, client, naan, nested_shoulder
    ) -> None:
        """The suffix is cut at the same significant character, not the raw one."""
        ark = self._create_ark(
            naan, nested_shoulder, "k3n7q9wb2", "https://example.com/item"
        )
        res = client.get(f"/ark:/{naan.naan}{nested_shoulder.shoulder}k3n7-q9wb2/v2")
        assert res.status_code == 302
        assert res.url == "https://example.com/item/v2"

    @pytest.mark.django_db
    def test_hyphen_in_the_qualifier_is_preserved(
        self, client, naan, nested_shoulder
    ) -> None:
        """A qualifier addresses a path this resolver does not assign.

        Its hyphens are therefore significant and must reach the target intact.
        """
        ark = self._create_ark(
            naan, nested_shoulder, "k3n7q9wb2", "https://example.com/item"
        )
        res = client.get(f"/ark:/{ark.ark}/scan-001.nxs")
        assert res.status_code == 302
        assert res.url == "https://example.com/item/scan-001.nxs"


class TestArkInvariant:
    """Test the invariant Ark.clean() enforces on the ark string."""

    @pytest.mark.django_db
    def test_created_ark_passes_clean(self, naan, shoulder) -> None:
        """Ark.clean() agrees with the ark string Ark.create() builds."""
        Ark.create(naan, shoulder).clean()

    @pytest.mark.django_db
    def test_clean_rejects_mismatched_ark(self, naan, shoulder) -> None:
        """Ark.clean() still catches an ark string that disagrees with its parts."""
        ark = Ark.create(naan, shoulder)
        ark.assigned_name = "tampered"
        with pytest.raises(ValidationError):
            ark.clean()
