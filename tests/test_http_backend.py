from __future__ import annotations

from openclaw_env.backend.multi_app_backend import MultiAppBackend


def test_curl_route_returns_mock_json_in_multi_mode(tmp_path):
    backend = MultiAppBackend(real_openclaw=False)
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli(
        'curl -sG "https://api.open-meteo.com/v1/forecast" '
        '--data-urlencode "latitude=40.7128" '
        '--data-urlencode "longitude=-74.0060" '
        '--data-urlencode "current=temperature_2m,relative_humidity_2m,weather_code" '
        '--data-urlencode "timezone=America/New_York"'
    )
    assert result.exit_code == 0
    assert '"current"' in result.stdout
    assert '"temperature_2m"' in result.stdout
    assert result.meta is not None
    assert result.meta.get("http_mode") == "mock"


def test_curl_geocoding_route_returns_mock_json_in_multi_mode(tmp_path):
    backend = MultiAppBackend(real_openclaw=False)
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli(
        'curl -sG "https://geocoding-api.open-meteo.com/v1/search" '
        '--data-urlencode "name=New York" '
        '--data-urlencode "count=1" '
        '--data-urlencode "language=en" '
        '--data-urlencode "format=json"'
    )
    assert result.exit_code == 0
    assert '"results"' in result.stdout
    assert '"timezone"' in result.stdout


def test_curl_route_rejects_unknown_endpoint_in_multi_mode(tmp_path):
    backend = MultiAppBackend(real_openclaw=False)
    backend.initialize(str(tmp_path), {})
    result = backend.execute_cli('curl -sG "https://example.com/unknown-endpoint"')
    assert result.exit_code != 0
    assert "Unsupported curl endpoint" in result.stderr
