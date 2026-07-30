"""Tests for ark/views.py, comprising the main endpoints for arklet."""

import uuid
from dataclasses import asdict, dataclass
from itertools import chain, count
from unittest.mock import patch

import pytest
from django.core.exceptions import ValidationError

from ark.models import Ark, Key, Naan, Shoulder
from ark.utils import parse_ark


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
    """Create a multi-segment shoulder, e.g. ark:/1/jc2/nii/<noid>.

    A trailing slash is what separates the shoulder from the generated name.
    """
    return Shoulder.objects.create(
        shoulder="/jc2/nii/", naan=naan, name="Nested", description="A Shoulder"
    )


@pytest.fixture
def other_naan(db):
    """Create a second NAAN, used to test cross-NAAN isolation."""
    return Naan.objects.create(
        naan=2, name="Other", description="Another NAAN", url="https://other.example.com"
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
