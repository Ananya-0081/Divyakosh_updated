import hashlib
import json
import requests

try:
    # Same stable ID the SQLite importer stores in donations.donation_id.
    # live_camera.py puts the project root on sys.path before importing this
    # module, so the `sqlite` package resolves.
    from sqlite.importer import make_donation_id
except Exception:
    make_donation_id = None


BLOCKCHAIN_API_URL = "http://localhost:3000/register-proof"


def create_proof_hash(record):
    """
    Create a deterministic SHA-256 fingerprint of a DaanDristi
    accepted donation record.

    This is the LEGACY raw-record hash. It is still sent to the API as a
    lookup key / fallback. When the API finds the record in SQLite it
    registers the CANONICAL AI event hash on Monad instead and returns it
    in the response.
    """

    canonical_data = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":")
    )

    proof_hash = "0x" + hashlib.sha256(
        canonical_data.encode("utf-8")
    ).hexdigest()

    return proof_hash


def get_donation_id(record):
    """
    Return the SQLite donation_id for this record, or None if it cannot
    be computed. Lets the API find the exact SQLite row.
    """

    if make_donation_id is None:
        return None

    try:
        denomination = int(str(record["predicted_denomination"]).strip())
        return make_donation_id(record, denomination)
    except (KeyError, TypeError, ValueError):
        return None


SUPPORTED_DENOMINATIONS = frozenset(["10", "20", "50", "100", "200", "500"])


def register_blockchain_proof(record):
    """
    Send an authorized accepted DivyaKosh record to the blockchain API.

    Returns the blockchain response, or None if registration fails or is rejected.
    """

    # Defense-in-depth: strict donation authorization validation
    if not isinstance(record, dict):
        print("Blockchain: skipped - record is not a dictionary.")
        return None

    if record.get("count_authorized") is not True:
        print("Blockchain: skipped - count_authorized is not True.")
        return None

    if record.get("operational_action") != "COUNT":
        print(f"Blockchain: skipped - operational_action is not COUNT ({record.get('operational_action')}).")
        return None

    if record.get("classification") != "ACCEPT_NOTE":
        print(f"Blockchain: skipped - classification is not ACCEPT_NOTE ({record.get('classification')}).")
        return None

    denom_str = str(record.get("predicted_denomination", "")).strip()
    if denom_str not in SUPPORTED_DENOMINATIONS:
        print(f"Blockchain: skipped - unsupported denomination '{denom_str}'.")
        return None

    # Blockchain payload requires legacy decision == "ACCEPT"
    if record.get("decision") != "ACCEPT":
        print("Blockchain: skipped because decision is not ACCEPT.")
        return None

    legacy_hash = create_proof_hash(record)
    donation_id = get_donation_id(record)

    print()
    print("=== BLOCKCHAIN REGISTRATION ===")
    print("Record hash (legacy):", legacy_hash)
    print("Donation ID:", donation_id)
    print("Source ID:", record.get("pocket_id"))

    payload = {
        "proofHash": legacy_hash,
        "sourceId": record.get("pocket_id", "unknown")
    }

    if donation_id:
        payload["donationId"] = donation_id

    try:
        response = requests.post(
            BLOCKCHAIN_API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        # The API returns the hash it actually registered
        # (canonical AI event hash when the SQLite row was found).
        registered_hash = result.get("proofHash") or legacy_hash

        print("Blockchain: proof registered successfully.")
        print("Proof scheme:", result.get("proofScheme"))
        print("Proof hash (registered):", registered_hash)
        print("Transaction:", result.get("transactionHash"))
        print("Block:", result.get("blockNumber"))

        return {
            "proof_hash": registered_hash,
            "legacy_proof_hash": legacy_hash,
            "proof_scheme": result.get("proofScheme"),
            "event_id": result.get("eventId"),
            "transaction_hash": result.get("transactionHash"),
            "block_number": result.get("blockNumber"),
            "source_id": record.get("pocket_id")
        }

    except requests.RequestException as error:
        print(f"Blockchain warning: registration failed ({error})")
        print("ML record remains safe in JSONL/SQLite.")
        return None
