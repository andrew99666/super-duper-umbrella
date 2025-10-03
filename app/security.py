"""Utilities for symmetric encryption of secrets at rest."""
from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.fernet import Fernet


class EncryptionManager:
    """Encrypts and decrypts text values using Fernet."""

    def __init__(self, secret_key: str) -> None:
        if not secret_key:
            raise ValueError("APP_SECRET_KEY must be set for encryption.")
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        fernet_key = base64.urlsafe_b64encode(digest)
        self._fernet = Fernet(fernet_key)

    def encrypt(self, value: str) -> bytes:
        return self._fernet.encrypt(value.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        return self._fernet.decrypt(token).decode("utf-8")


@lru_cache(maxsize=1)
def get_encryption_manager() -> EncryptionManager:
    secret = os.getenv("APP_SECRET_KEY")
    return EncryptionManager(secret_key=secret or "")
