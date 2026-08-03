import json
import logging
import os

from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models.functions import Length
from django.http import (
    Http404,
    HttpRequest,
    HttpResponseBadRequest,
    HttpResponseForbidden,
    HttpResponseNotAllowed,
    HttpResponseRedirect,
    HttpResponseServerError,
    JsonResponse, HttpResponseNotFound,
)
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from ark.forms import MintArkForm, UpdateArkForm
from ark.models import Ark, Naan, Key, Shoulder
from ark.utils import (
    gen_prefixes,
    parse_ark,
    parse_ark_lookup,
    split_after_normalized,
    strip_hyphens,
)
from arklet.settings import env

COLLISIONS = 10

logger = logging.getLogger(__name__)

def authorize(request, naan):
    bearer_token = request.headers.get("Authorization")
    if not bearer_token:
        return None

    key = bearer_token.split()[-1]

    try:
        keys = Key.objects.filter(naan=naan,active=True)
        for k in keys:
            if k.check_password(key):
                return k.naan
        return None
    except ValidationError as e:  # probably an invalid key
        return None
    

@csrf_exempt
def mint_ark(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(permitted_methods=["POST"])

    try:
        unsafe_mint_request = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, TypeError) as e:
        return HttpResponseBadRequest(e)

    mint_request = MintArkForm(unsafe_mint_request)

    if not mint_request.is_valid():
        return JsonResponse(mint_request.errors, status=400)

    # Pop these keys so that we can pass the cleaned data
    # dict directly to the create method later
    naan = mint_request.cleaned_data.pop("naan")
    authorized_naan = authorize(request, naan)
    if authorized_naan is None:
        return HttpResponseForbidden()

    shoulder = mint_request.cleaned_data.pop("shoulder")
    shoulder_obj = Shoulder.objects.filter(
        shoulder=shoulder, naan=authorized_naan
    ).first()
    if shoulder_obj is None:
        return HttpResponseBadRequest(f"Shoulder {shoulder} does not exist")

    ark, collisions = None, 0
    for _ in range(COLLISIONS):
        candidate = Ark.create(authorized_naan, shoulder_obj)
        candidate.set_fields(mint_request.cleaned_data)
        try:
            # force_insert, or saving over an existing name would silently
            # overwrite that ark instead of registering the collision.
            candidate.save(force_insert=True)
        except IntegrityError:
            collisions += 1
            continue
        ark = candidate
        break

    if not ark:
        msg = f"Gave up creating ark after {collisions} collision(s)"
        logger.error(msg)
        return HttpResponseServerError(msg)
    if ark and collisions > 0:
        logger.warning("Ark created after %d collision(s)", collisions)

    return JsonResponse({"ark": str(ark)})


@csrf_exempt
def update_ark(request):
    if request.method != "PUT":
        return HttpResponseNotAllowed(permitted_methods=["PUT"])

    try:
        unsafe_update_request = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, TypeError) as e:
        return HttpResponseBadRequest(e)

    # TODO: test input data with wrong structure
    update_request = UpdateArkForm(unsafe_update_request)

    if not update_request.is_valid():
        return JsonResponse(update_request.errors, status=400)

    # The ark field is immutable, pop it out of the cleaned
    # data dictionary here so we don't try to update it later
    ark = update_request.cleaned_data.pop("ark")

    _, naan, assigned_name = parse_ark(ark)

    authorized_naan = authorize(request, naan)
    if authorized_naan is None:
        return HttpResponseForbidden()

    try:
        ark_obj = Ark.objects.get(ark=f"{naan}/{assigned_name}")
    except Ark.DoesNotExist:
        raise Http404
    
    ark_obj.set_fields(update_request.cleaned_data)
    ark_obj.save()

    return JsonResponse(ark_to_json(ark_obj, metadata=False))


def resolve_ark(request, ark: str):
    info_inflection = 'info' in request.GET
    json_inflection = 'json' in request.GET

    try:
        _, naan, identifier = parse_ark(ark)
    except ValueError as e:
        return HttpResponseBadRequest(e)

    ark_str = f"{naan}/{identifier}"
    # Hyphens are insignificant, so an ark that picked one up in transit still
    # has to find its record. Stored arks are matched first so that a legacy
    # record containing a hyphen keeps resolving.
    normalized = strip_hyphens(identifier)
    ark_obj = Ark.objects.filter(ark=ark_str).first()
    if ark_obj is None and normalized != identifier:
        ark_obj = Ark.objects.filter(ark=f"{naan}/{normalized}").first()
    if ark_obj:
        if info_inflection:
            return view_ark(request, ark_obj)
        if json_inflection:
            return json_ark(request, ark_obj)
        if not ark_obj.url:
            return view_ark(request, ark_obj)
        return HttpResponseRedirect(ark_obj.url + '?' + request.META['QUERY_STRING'])
    else:
        # Ark not found. Try to find an ark that is a prefix.
        prefixes = [f"{naan}/{a}" for a in gen_prefixes(normalized)]
        # Get the one with the longest prefix, so that a nested ark (e.g. an
        # item under a collection) wins over its shorter ancestors.
        ark_prefix = (
            Ark.objects.filter(ark__in=prefixes)
            .order_by(Length('ark').desc())
            .first()
        )
        if ark_prefix:
            # The ancestor was matched without hyphens, so measure it that way
            # and cut the requested string at the same significant character.
            ancestor_name = ark_prefix.ark.removeprefix(f"{naan}/")
            _, suffix = split_after_normalized(
                identifier, len(strip_hyphens(ancestor_name))
            )
            # An inflection asks this resolver for metadata, and arklet is the
            # end of the chain: there is no downstream service to forward the
            # request to. Answer with the nearest registered ancestor's
            # metadata, labelled with the ark that was actually requested, so
            # that metadata stays available at sub-resource granularity even
            # though only the ancestor was minted.
            if info_inflection:
                return view_ark(request, ark_prefix, requested_ark=ark_str, suffix=suffix)
            if json_inflection:
                return json_ark(request, ark_prefix, requested_ark=ark_str, suffix=suffix)
            if not ark_prefix.url:
                # No target to pass the suffix through to, so describe the
                # nearest registered ancestor instead of redirecting to a
                # relative path built from an empty url.
                return view_ark(request, ark_prefix, requested_ark=ark_str, suffix=suffix)
            return HttpResponseRedirect(ark_prefix.url + suffix)
        else:
            if info_inflection or json_inflection:
                return HttpResponseNotFound(f"ark:{ark_str} metadata is unknown to this resolver.")
            try:
                naan_obj = Naan.objects.get(naan=naan)
                if naan_obj.url.endswith(env("ARKLET_HOST")):
                    return HttpResponseNotFound(f"ark:{ark_str} is unknown to this resolver.")
                return HttpResponseRedirect(
                    f"{naan_obj.url}/ark:/{ark_str}"
                )
            except Naan.DoesNotExist:
                return HttpResponseNotFound(f"naan {naan} is unknown to this resolver.")

def resolved_url(ark: Ark, suffix: str = "") -> str:
    """Where the requested ark resolves to.

    For a suffixed request this is the ancestor's url with the suffix appended,
    which is exactly the Location a plain (uninflected) request would receive.
    An ark reserved without a url keeps an empty url rather than degrading to a
    bare suffix.
    """
    if suffix and ark.url:
        return f"{ark.url}{suffix}"
    return ark.url


"""
Return HTML human readable webpage information about the Ark object
"""
def view_ark(request: HttpRequest, ark: Ark, requested_ark: str = None, suffix: str = ""):

    context = {
        'ark': requested_ark or ark.ark,
        'url': resolved_url(ark, suffix),
        'label': ark.title,
        'type': ark.type,
        'commitment': ark.commitment,
        'identifier': ark.identifier,
        'format': ark.format,
        'relation': ark.relation,
        'source': ark.source,
        'metadata': ark.metadata,
        'inherited_from': ark.ark if suffix else "",
        'suffix': suffix,
    }

    return render(request, 'info.html', context)

"""
Return the Ark object as JSON
"""
def ark_to_json(ark: Ark, metadata=True, requested_ark: str = None, suffix: str = ""):
    """Serialize an ark, optionally as the answer for a suffixed descendant.

    When `suffix` is set the record describes `requested_ark` while the values
    are inherited from `ark`, the nearest registered ancestor. `inherited_from`
    and `suffix` are only present in that case, so exact-match responses and the
    mint/update/bulk payloads keep their existing shape.
    """
    data = {
        'ark': requested_ark or ark.ark,
        'url': resolved_url(ark, suffix),
        'title': ark.title,
        'type': ark.type,
        'commitment': ark.commitment,
        'identifier': ark.identifier,
        'format': ark.format,
        'relation': ark.relation,
        'source': ark.source,
        'metadata': ark.metadata
    }
    if suffix:
        data['inherited_from'] = ark.ark
        data['suffix'] = suffix
    if not metadata:
        return data
    obj = {}
    for key in data:
        # Copy: COLUMN_METADATA is class level, so writing 'value' into the
        # dict it returns would leak this ark's values into every later
        # response.
        obj[key] = dict(Ark.COLUMN_METADATA.get(key, {}))
        obj[key]['value'] = data[key]
    return obj

def json_ark(request: HttpRequest, ark: Ark, requested_ark: str = None, suffix: str = ""):
    obj = ark_to_json(ark, requested_ark=requested_ark, suffix=suffix)
    # Return the JSON response
    return JsonResponse(obj)

@csrf_exempt
def batch_query_arks(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, TypeError) as e:
        return HttpResponseBadRequest(e)
    if len(data) > 100:
        return HttpResponseBadRequest("Exceeded max rows (100)")
    arks = [parse_ark_lookup(d.get('ark')) for d in data]
    ark_objs = Ark.objects.filter(ark__in=arks)
    resp = [ark_to_json(ark, metadata=False) for ark in ark_objs]
    return JsonResponse(resp, safe=False)


@csrf_exempt
def batch_update_arks(request):
    try:
        data = json.loads(request.body.decode("utf-8"))['data']
    except (json.JSONDecodeError, TypeError) as e:
        return HttpResponseBadRequest(e)
    if len(data) > 100:
        return HttpResponseBadRequest("Exceeded max rows (100)")
    
    naans = set()
    for d in data:
        if 'ark' not in d:
            return HttpResponseBadRequest("Each record must have an 'ark' field.")
        _, naan, _ = parse_ark(d['ark'])
        naans.add(naan)

    if len(naans) != 1:
        return HttpResponseBadRequest("Batch queries are limited to one NAAN at a time")
    
    naan = naans.pop()
    authorized_naan = authorize(request, naan)
    if authorized_naan is None:
        return HttpResponseForbidden()

    
    arks = [parse_ark_lookup(d.get('ark')) for d in data]
    ark_objs = Ark.objects.filter(ark__in=arks)
    
    # track the fields we have seen so far for efficient updating
    seen_fields = set()
    for ark_obj, new_record in zip(ark_objs, data):
        ark_obj.set_fields(new_record)
        seen_fields.update(new_record.keys())
    # don't update primary key
    seen_fields.remove('ark')
    n_updated = Ark.objects.bulk_update(ark_objs, fields=seen_fields)
    return JsonResponse({
        'num_received': len(data),
        'num_updated': n_updated
    })

@csrf_exempt
def batch_mint_arks(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, TypeError) as e:
        return HttpResponseBadRequest(e)

    naan = data.get('naan')
    authorized_naan = authorize(request, naan)
    if authorized_naan is None:
        return HttpResponseForbidden()
    records = data['data']

    if len(records) > 100:
        return HttpResponseBadRequest("Exceeded max rows (100)")

    shoulders = set()
    for d in records:
        if 'shoulder' not in d:
            return HttpResponseBadRequest("shoulder value must be present in every record")
        shoulders.add(d['shoulder'])
    shoulder_objs = dict()
    for s in shoulders:
        shoulder_obj = Shoulder.objects.filter(shoulder=s, naan=authorized_naan).first()
        if shoulder_obj is None:
            return HttpResponseBadRequest(f"shoulder {s} does not exist")
        shoulder_objs[s] = shoulder_obj

    created = None
    for _ in range(COLLISIONS):
        # Attempt to mint the batch with max COLLISION retries times
        try:
            new_arks = []
            for record in records:
                shoulder = shoulder_objs[record['shoulder']]
                new_ark = Ark.create(authorized_naan, shoulder)
                new_ark.set_fields(record)
                new_arks.append(new_ark)
            created = Ark.objects.bulk_create(new_arks)
        except IntegrityError:
            continue
        break
    else:
        msg = f"Gave up creating bulk arks after {COLLISIONS} collision(s)"
        logger.error(msg)
        return HttpResponseServerError(msg)
    return JsonResponse({
        'num_received': len(records),
        'arks_created': [ark_to_json(c, metadata=False) for c in created]
    })

def status(request):
    service = 'resolver' if os.environ.get("RESOLVER") else 'minter'

    return JsonResponse({
        'service': service,
        'status': 'ok!',
    })