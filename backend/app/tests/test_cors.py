def test_vercel_frontend_origin_is_allowed(client):
    response = client.options(
        "/api/auth/request-code",
        headers={
            "Origin": "https://data-collect-web.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://data-collect-web.vercel.app"
