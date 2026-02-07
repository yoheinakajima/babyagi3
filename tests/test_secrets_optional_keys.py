from tools.secrets import _KEYRING_KNOWN_KEYS


def test_optional_api_keys_are_loaded_from_keyring_on_startup():
    expected = {
        "PEOPLEDATALABS_API_KEY",
        "VOILANORBERT_API_KEY",
        "HUNTER_API_KEY",
        "EXA_API_KEY",
        "HAPPENSTANCE_API_KEY",
        "X_API_BEARER_TOKEN",
        "RUNWAY_API_KEY",
        "ELEVENLABS_API_KEY",
        "VIDEODB_API_KEY",
        "GODADDY_API_KEY",
        "GODADDY_API_SECRET",
        "SHOPIFY_ACCESS_TOKEN",
        "SHOPIFY_STORE_DOMAIN",
        "PRINTFUL_API_KEY",
        "GITHUB_TOKEN",
    }
    assert expected.issubset(set(_KEYRING_KNOWN_KEYS))
