import logging
import os
import secrets
import smtplib
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import EmailLoginCode, UserAccount, UserSessionToken

logger = logging.getLogger(__name__)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def create_login_code(db: Session, email: str) -> str:
    normalized = normalize_email(email)
    account = db.get(UserAccount, normalized)
    if not account:
        account = UserAccount(email=normalized, verified=False)
        db.add(account)

    code = generate_code()
    login_code = EmailLoginCode(
        email=normalized,
        code=code,
        expires_at_utc=datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10),
    )
    db.add(login_code)
    db.commit()
    return code


def smtp_configured() -> bool:
    return all(
        os.getenv(name)
        for name in ["SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL"]
    )


def smtp_timeout_seconds() -> float:
    try:
        return float(os.getenv("SMTP_TIMEOUT_SECONDS", "8"))
    except ValueError:
        return 8.0


def send_login_code(email: str, code: str) -> bool:
    if not smtp_configured():
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    username = os.environ["SMTP_USERNAME"]
    password = os.environ["SMTP_PASSWORD"]
    from_email = os.environ["SMTP_FROM_EMAIL"]
    use_ssl = os.getenv("SMTP_USE_SSL", "true").lower() in {"true", "1", "yes"}
    timeout = smtp_timeout_seconds()

    message = EmailMessage()
    message["Subject"] = "Vigil Recorder login code"
    message["From"] = from_email
    message["To"] = email
    message.set_content(f"Your Vigil Recorder login code is: {code}\n\nThis code expires in 10 minutes.")

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=timeout) as smtp:
                smtp.login(username, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                smtp.starttls()
                smtp.login(username, password)
                smtp.send_message(message)
    except (OSError, TimeoutError, smtplib.SMTPException):
        logger.warning("Failed to send login code via SMTP", exc_info=True)
        return False
    return True


def verify_login_code(db: Session, email: str, code: str) -> bool:
    normalized = normalize_email(email)
    now = datetime.now(UTC).replace(tzinfo=None)
    login_code = (
        db.execute(
            select(EmailLoginCode)
            .where(
                EmailLoginCode.email == normalized,
                EmailLoginCode.code == code.strip(),
                EmailLoginCode.used_at_utc.is_(None),
                EmailLoginCode.expires_at_utc >= now,
            )
            .order_by(EmailLoginCode.created_at_utc.desc())
        )
        .scalars()
        .first()
    )
    if not login_code:
        return False

    login_code.used_at_utc = now
    account = db.get(UserAccount, normalized)
    if not account:
        account = UserAccount(email=normalized)
        db.add(account)
    account.verified = True
    account.last_login_at_utc = now
    db.commit()
    return True


def create_session_token(db: Session, email: str) -> tuple[str, datetime]:
    normalized = normalize_email(email)
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)
    db.add(UserSessionToken(token=token, email=normalized, expires_at_utc=expires_at))
    db.commit()
    return token, expires_at


def verify_session_token(db: Session, email: str, token: str | None) -> bool:
    if not token:
        return False
    normalized = normalize_email(email)
    now = datetime.now(UTC).replace(tzinfo=None)
    session_token = db.get(UserSessionToken, token)
    return bool(
        session_token
        and session_token.email == normalized
        and session_token.expires_at_utc >= now
    )
