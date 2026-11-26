"""A JWT secret persistence failure must be visible to the operator."""

import logging
from pathlib import Path
from unittest.mock import patch

from security.auth_system import SimpleJWT


def test_secret_read_failure_logs_token_invalidation_warning(tmp_path, caplog) -> None:
    secret_file = tmp_path / ".jwt_secret"
    secret_file.write_text("existing-secret", encoding="utf-8")

    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        with caplog.at_level(logging.ERROR):
            jwt = SimpleJWT(secret_file=str(secret_file))

    assert jwt.secret_key
    assert "JWT_SECRET_KEY" in caplog.text
    assert "permission denied" in caplog.text
