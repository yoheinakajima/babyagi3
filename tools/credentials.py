"""
Credential Management Tools

Secure storage and retrieval of user accounts and payment methods.

Credentials are stored in two layers:
1. Metadata (service, username, etc.) - in SQLite database
2. Sensitive data (passwords, card numbers) - in keyring/secrets manager

This separation ensures:
- Quick lookup of what accounts exist
- Secure storage of actual secrets
- Persistence across sessions
"""

import re
from tools import tool


def _generate_secret_ref(service: str, credential_type: str, field: str) -> str:
    """Generate a reference key for storing secrets in keyring."""
    # Normalize service name for use as key
    safe_service = re.sub(r'[^a-zA-Z0-9_-]', '_', service.lower())
    return f"cred_{safe_service}_{credential_type}_{field}"


def _detect_card_type(card_number: str) -> str:
    """Detect credit card type from card number."""
    # Remove any spaces or dashes
    num = re.sub(r'[\s-]', '', card_number)

    if num.startswith('4'):
        return 'visa'
    elif num.startswith(('51', '52', '53', '54', '55')):
        return 'mastercard'
    elif len(num) >= 4:
        try:
            if 2221 <= int(num[:4]) <= 2720:
                return 'mastercard'
        except ValueError:
            pass
    if num.startswith(('34', '37')):
        return 'amex'
    elif num.startswith('6011') or num.startswith('65'):
        return 'discover'
    elif num.startswith(('62', '88')):
        return 'unionpay'
    return 'unknown'


@tool(packages=["keyring"])
def store_credential(
    service: str,
    credential_type: str = "account",
    username: str = None,
    email: str = None,
    password: str = None,
    card_number: str = None,
    card_expiry: str = None,
    card_cvv: str = None,
    billing_name: str = None,
    billing_address: str = None,
    notes: str = None,
    agent=None,
) -> dict:
    """Store a credential (user account or credit card) securely.

    ALWAYS use this when creating accounts or saving payment methods.
    Sensitive data (passwords, card numbers) are stored encrypted in keyring.

    For USER ACCOUNTS:
        store_credential(
            service="yohei.ai",
            username="user@email.com",
            password="the_password"
        )

    For CREDIT CARDS:
        store_credential(
            service="payment",
            credential_type="credit_card",
            card_number="4111111111111111",
            card_expiry="12/25",
            card_cvv="123",
            billing_name="John Doe"
        )

    Args:
        service: Service/website name (e.g., "yohei.ai", "github.com")
        credential_type: "account" or "credit_card"
        username: Username for account
        email: Email for account
        password: Password (stored securely in keyring)
        card_number: Full card number (only last 4 stored in DB, full in keyring)
        card_expiry: Expiry date MM/YY
        card_cvv: CVV (stored in keyring only)
        billing_name: Name on card
        billing_address: Billing address
        notes: Additional notes
    """
    try:
        import keyring
        from tools.secrets import KEYRING_SERVICE
    except ImportError:
        return {"error": "keyring not available - cannot store credentials securely"}

    password_ref = None
    card_ref = None
    card_last_four = None
    card_type = None

    # Store password in keyring if provided
    if password:
        password_ref = _generate_secret_ref(service, credential_type, "password")
        keyring.set_password(KEYRING_SERVICE, password_ref, password)

    # Store card details in keyring if provided
    if card_number:
        # Store full card number
        card_ref = _generate_secret_ref(service, credential_type, "card")
        keyring.set_password(KEYRING_SERVICE, card_ref, card_number)

        # Extract last 4 digits and detect type
        clean_num = re.sub(r'[\s-]', '', card_number)
        card_last_four = clean_num[-4:]
        card_type = _detect_card_type(clean_num)

        # Store CVV if provided
        if card_cvv:
            cvv_ref = _generate_secret_ref(service, credential_type, "cvv")
            keyring.set_password(KEYRING_SERVICE, cvv_ref, card_cvv)

    # Also store metadata in memory database if available
    db_stored = False
    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            agent.memory.store.store_credential(
                service=service,
                credential_type=credential_type,
                username=username,
                email=email,
                password_ref=password_ref,
                card_last_four=card_last_four,
                card_type=card_type,
                card_expiry=card_expiry,
                card_ref=card_ref,
                billing_name=billing_name,
                billing_address=billing_address,
                notes=notes,
            )
            db_stored = True
        except Exception as e:
            # Don't fail if DB storage fails - keyring storage is primary
            pass

    return {
        "stored": True,
        "service": service,
        "credential_type": credential_type,
        "username": username,
        "email": email,
        "password_stored": password_ref is not None,
        "card_last_four": card_last_four,
        "card_type": card_type,
        "billing_name": billing_name,
        "db_stored": db_stored,
        "message": f"Credential for {service} stored securely. Use get_credential('{service}') to retrieve."
    }


@tool(packages=["keyring"])
def get_credential(
    service: str,
    credential_type: str = None,
    include_secrets: bool = False,
    agent=None,
) -> dict:
    """Retrieve a stored credential.

    By default, returns metadata only (service, username, etc.).
    Set include_secrets=True to also retrieve passwords/card numbers.

    Args:
        service: Service name to look up
        credential_type: Optional filter ("account" or "credit_card")
        include_secrets: If True, also retrieve password/card from keyring

    Returns:
        Credential details including username, email, and optionally password
    """
    try:
        import keyring
        from tools.secrets import KEYRING_SERVICE
    except ImportError:
        return {"error": "keyring not available"}

    result = {
        "service": service,
        "found": False,
    }

    # First, try to get metadata from memory store
    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            cred = agent.memory.store.get_credential(service, credential_type)
            if cred:
                result["found"] = True
                result["credential_type"] = cred.credential_type
                result["username"] = cred.username
                result["email"] = cred.email
                result["card_last_four"] = cred.card_last_four
                result["card_type"] = cred.card_type
                result["card_expiry"] = cred.card_expiry
                result["billing_name"] = cred.billing_name
                result["notes"] = cred.notes

                # Get secrets from keyring if requested
                if include_secrets:
                    if cred.password_ref:
                        password = keyring.get_password(KEYRING_SERVICE, cred.password_ref)
                        if password:
                            result["password"] = password
                    if cred.card_ref:
                        card = keyring.get_password(KEYRING_SERVICE, cred.card_ref)
                        if card:
                            result["card_number"] = card
                        cvv_ref = _generate_secret_ref(service, "credit_card", "cvv")
                        cvv = keyring.get_password(KEYRING_SERVICE, cvv_ref)
                        if cvv:
                            result["card_cvv"] = cvv

                # Update last used timestamp
                agent.memory.store.update_credential_last_used(cred.id)
                return result
        except Exception:
            pass  # Fall back to keyring-only lookup

    # Fallback: check keyring directly
    password_ref = _generate_secret_ref(service, credential_type or "account", "password")
    card_ref = _generate_secret_ref(service, "credit_card", "card")

    # Check for password
    password = keyring.get_password(KEYRING_SERVICE, password_ref)
    has_password = password is not None

    # Check for card
    card = keyring.get_password(KEYRING_SERVICE, card_ref)
    has_card = card is not None

    if has_password or has_card:
        result["found"] = True

        if has_password:
            result["credential_type"] = "account"
            result["has_password"] = True
            if include_secrets:
                result["password"] = password

        if has_card:
            result["credential_type"] = "credit_card"
            result["has_card"] = True
            if include_secrets:
                result["card_number"] = card
                # Also get CVV if available
                cvv_ref = _generate_secret_ref(service, "credit_card", "cvv")
                cvv = keyring.get_password(KEYRING_SERVICE, cvv_ref)
                if cvv:
                    result["card_cvv"] = cvv

    if not result["found"]:
        result["message"] = f"No credential found for '{service}'"

    return result


@tool
def list_credentials(agent=None) -> dict:
    """List all stored credentials (without sensitive data).

    Returns a summary of all accounts and payment methods that have been stored.
    Use get_credential(service, include_secrets=True) to retrieve actual passwords.
    """
    credentials = []

    # First, try to get from memory store (preferred - better metadata)
    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            creds = agent.memory.store.list_credentials()
            for cred in creds:
                cred_info = {
                    "service": cred.service,
                    "credential_type": cred.credential_type,
                }
                if cred.username:
                    cred_info["username"] = cred.username
                if cred.email:
                    cred_info["email"] = cred.email
                if cred.card_last_four:
                    cred_info["card_last_four"] = f"****{cred.card_last_four}"
                    cred_info["card_type"] = cred.card_type
                credentials.append(cred_info)

            if credentials:
                return {
                    "count": len(credentials),
                    "credentials": credentials,
                    "note": "Use get_credential(service, include_secrets=True) to retrieve passwords/card numbers"
                }
        except Exception:
            pass  # Fall back to keyring-only

    # Fallback: scan keyring
    try:
        from tools.secrets import list_secrets

        secrets = list_secrets()
        seen_services = set()

        for secret in secrets.get("secrets", []):
            name = secret.get("name", "")

            # Parse credential references (format: cred_service_type_field)
            if name.startswith("cred_"):
                parts = name.split("_")
                if len(parts) >= 4:
                    service = parts[1]
                    cred_type = parts[2]

                    key = f"{service}_{cred_type}"
                    if key not in seen_services:
                        seen_services.add(key)
                        credentials.append({
                            "service": service,
                            "credential_type": cred_type,
                        })
    except Exception:
        pass

    return {
        "count": len(credentials),
        "credentials": credentials,
        "note": "Use get_credential(service, include_secrets=True) to retrieve actual passwords/card numbers"
    }


@tool
def search_credentials(query: str, agent=None) -> dict:
    """Search credentials by service name, username, or email.

    Args:
        query: Search query (partial match on service, username, email)

    Returns:
        List of matching credentials
    """
    credentials = []

    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            creds = agent.memory.store.search_credentials(query)
            for cred in creds:
                cred_info = {
                    "service": cred.service,
                    "credential_type": cred.credential_type,
                }
                if cred.username:
                    cred_info["username"] = cred.username
                if cred.email:
                    cred_info["email"] = cred.email
                if cred.card_last_four:
                    cred_info["card_last_four"] = f"****{cred.card_last_four}"
                credentials.append(cred_info)

            return {
                "query": query,
                "count": len(credentials),
                "credentials": credentials,
            }
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    # Fallback if no memory store
    return {
        "query": query,
        "count": 0,
        "credentials": [],
        "note": "Memory store not available - use list_credentials for keyring scan"
    }


@tool(packages=["keyring"])
def delete_credential(service: str, credential_type: str = None, agent=None) -> dict:
    """Delete a stored credential.

    Removes both the metadata and the sensitive data from keyring.

    Args:
        service: Service name
        credential_type: "account" or "credit_card" (if not specified, deletes both)
    """
    try:
        import keyring
        from tools.secrets import KEYRING_SERVICE
    except ImportError:
        return {"error": "keyring not available"}

    deleted = []
    types_to_check = [credential_type] if credential_type else ["account", "credit_card"]

    # Delete from memory store first
    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            cred = agent.memory.store.get_credential(service, credential_type)
            if cred:
                agent.memory.store.delete_credential(cred.id)
                deleted.append("database")
        except Exception:
            pass

    # Delete from keyring
    for ctype in types_to_check:
        # Try to delete password
        password_ref = _generate_secret_ref(service, ctype, "password")
        try:
            keyring.delete_password(KEYRING_SERVICE, password_ref)
            deleted.append(f"{ctype}:password")
        except Exception:
            pass

        # Try to delete card
        card_ref = _generate_secret_ref(service, ctype, "card")
        try:
            keyring.delete_password(KEYRING_SERVICE, card_ref)
            deleted.append(f"{ctype}:card")
        except Exception:
            pass

        # Try to delete CVV
        cvv_ref = _generate_secret_ref(service, ctype, "cvv")
        try:
            keyring.delete_password(KEYRING_SERVICE, cvv_ref)
            deleted.append(f"{ctype}:cvv")
        except Exception:
            pass

    if deleted:
        return {"deleted": True, "service": service, "removed": deleted}
    return {"deleted": False, "service": service, "message": "No credentials found"}
