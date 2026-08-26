from unittest import mock
import requests

from utils.messaging import send_telegram, mask_token

def test_mask_token():
    assert mask_token("") == "****"
    assert mask_token("short") == "****"
    assert mask_token("1234567890123456") == "123456...123456"

@mock.patch("utils.messaging.requests.post")
def test_send_telegram_success(mock_post):
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_post.return_value = mock_resp

    assert send_telegram("hello", "token", "chat") is True
    mock_post.assert_called_once()

def test_send_telegram_no_token():
    assert send_telegram("hello", "", "chat") is False
    assert send_telegram("hello", "token", "") is False

@mock.patch("utils.messaging.requests.post")
def test_send_telegram_truncate(mock_post):
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_post.return_value = mock_resp

    long_msg = "A" * 5000
    assert send_telegram(long_msg, "token", "chat") is True

    called_text = mock_post.call_args[1]["json"]["text"]
    assert len(called_text) == 4096
    assert called_text.endswith("… [truncated]")

@mock.patch("utils.messaging.time.sleep")
@mock.patch("utils.messaging.requests.post")
def test_send_telegram_429_retry(mock_post, mock_sleep):
    mock_429 = mock.Mock()
    mock_429.ok = False
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "1.5"}

    mock_200 = mock.Mock()
    mock_200.ok = True

    mock_post.side_effect = [mock_429, mock_200]

    assert send_telegram("hello", "token", "chat") is True
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(1.5)

@mock.patch("utils.messaging.time.sleep")
@mock.patch("utils.messaging.requests.post")
def test_send_telegram_429_retry_json(mock_post, mock_sleep):
    mock_429 = mock.Mock()
    mock_429.ok = False
    mock_429.status_code = 429
    mock_429.headers = {}
    mock_429.json.return_value = {"parameters": {"retry_after": 2.5}}

    mock_200 = mock.Mock()
    mock_200.ok = True

    mock_post.side_effect = [mock_429, mock_200]

    assert send_telegram("hello", "token", "chat") is True
    mock_sleep.assert_called_once_with(2.5)

@mock.patch("utils.messaging.time.sleep")
@mock.patch("utils.messaging.requests.post")
def test_send_telegram_429_fallback(mock_post, mock_sleep):
    mock_429 = mock.Mock()
    mock_429.ok = False
    mock_429.status_code = 429
    mock_429.headers = {}
    mock_429.json.return_value = {}

    mock_500 = mock.Mock()
    mock_500.ok = False
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error"

    mock_post.side_effect = [mock_429, mock_500]

    assert send_telegram("hello", "token", "chat") is False
    mock_sleep.assert_called_once_with(2.0)

@mock.patch("utils.messaging.time.sleep")
@mock.patch("utils.messaging.requests.post")
def test_send_telegram_network_error(mock_post, mock_sleep):
    mock_post.side_effect = requests.exceptions.ConnectionError("net error")

    assert send_telegram("hello", "token", "chat") is False
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(2)

@mock.patch("utils.messaging.requests.post")
def test_send_telegram_other_error(mock_post):
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden"

    mock_post.return_value = mock_resp
    assert send_telegram("hello", "token", "chat") is False
    mock_post.assert_called_once()
