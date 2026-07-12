from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.config.settings import settings

# tokenUrl tells Swagger where to get a token (our /token login endpoint).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(subject):
    """Issue a signed JWT for `subject` (the username), with an expiry."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token):
    """Verify signature + expiry and return the payload (raises on failure)."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def authenticate(username, password):
    """DEMO credential check. Production would delegate to an IdP (OIDC)."""
    return username == settings.auth_username and password == settings.auth_password


def get_current_user(token: str = Depends(oauth2_scheme)):
    """FastAPI dependency: require a valid Bearer token, return the username.

    Any protected route adds `Depends(get_current_user)` and automatically
    returns 401 for a missing/invalid/expired token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise credentials_exception
    subject = payload.get("sub")
    if subject is None:
        raise credentials_exception
    return subject
