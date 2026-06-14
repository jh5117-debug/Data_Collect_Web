import smtplib

from app.routes import auth as auth_routes
from app.services.email_auth import send_login_code


def test_request_code_returns_dev_code_when_email_delivery_fails(client, monkeypatch):
    monkeypatch.setattr(auth_routes, "send_login_code", lambda email, code: False)

    response = client.post("/api/auth/request-code", json={"email": "speaker@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "dev_code"
    assert len(body["dev_code"]) == 6


def test_send_login_code_fails_fast_on_smtp_error(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "sender@example.com")

    class FailingSMTP:
        def __init__(self, *args, **kwargs):
            raise smtplib.SMTPConnectError(421, "connection failed")

    monkeypatch.setattr(smtplib, "SMTP_SSL", FailingSMTP)

    assert send_login_code("speaker@example.com", "123456") is False
