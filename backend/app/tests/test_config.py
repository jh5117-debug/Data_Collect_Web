from app.config import _database_url_with_defaults


def test_supabase_database_url_adds_sslmode_require():
    url = (
        "postgresql+psycopg://postgres.example:password"
        "@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
    )

    assert _database_url_with_defaults(url).endswith("/postgres?sslmode=require")


def test_supabase_database_url_preserves_existing_sslmode():
    url = (
        "postgresql+psycopg://postgres.example:password"
        "@aws-1-us-east-1.pooler.supabase.com:5432/postgres?sslmode=verify-full"
    )

    assert _database_url_with_defaults(url).endswith("/postgres?sslmode=verify-full")
