from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from aider_aid.paths import get_ollama_servers_file

AIDER_AID_OLLAMA_SERVER_KEY = "aider-aid-ollama-server"


class OllamaServerError(Exception):
    """Base Ollama server store error."""


class OllamaServerNotFoundError(OllamaServerError):
    """Raised when a named server is missing."""


@dataclass(frozen=True)
class OllamaServer:
    name: str
    url: str


class OllamaServerStore:
    def __init__(self, *, config_root: Path | None = None) -> None:
        self._servers_file = get_ollama_servers_file(config_root)

    @property
    def servers_file(self) -> Path:
        return self._servers_file

    def _default_data(self) -> dict[str, object]:
        return {"version": 1, "servers": []}

    def _validate_url(self, url: str) -> str:
        normalized = url.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OllamaServerError(f'Invalid Ollama server URL "{url}". Use http(s)://host[:port].')
        return normalized

    def _load_data(self) -> dict[str, object]:
        if not self._servers_file.exists():
            return self._default_data()
        try:
            data = json.loads(self._servers_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OllamaServerError(f"Invalid JSON in {self._servers_file}: {exc}") from exc
        except OSError as exc:
            raise OllamaServerError(f"Unable to read {self._servers_file}: {exc}") from exc

        if not isinstance(data, dict):
            raise OllamaServerError("Ollama servers config must be a JSON object.")
        servers = data.get("servers", [])
        if not isinstance(servers, list):
            raise OllamaServerError("Ollama servers config 'servers' must be a list.")
        normalized_servers: list[dict[str, str]] = []
        for item in servers:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            url = item.get("url")
            if isinstance(name, str) and isinstance(url, str):
                normalized_servers.append({"name": name, "url": url})
        return {"version": 1, "servers": normalized_servers}

    def _write_data(self, data: dict[str, object]) -> None:
        self._servers_file.parent.mkdir(parents=True, exist_ok=True)
        self._servers_file.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def list_servers(self) -> list[OllamaServer]:
        data = self._load_data()
        items = data.get("servers", [])
        servers: list[OllamaServer] = []
        if not isinstance(items, list):
            return servers
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            url = item.get("url")
            if isinstance(name, str) and isinstance(url, str):
                servers.append(OllamaServer(name=name, url=url))
        return sorted(servers, key=lambda server: server.name.lower())

    def get_server(self, name: str) -> OllamaServer:
        lookup = name.strip()
        for server in self.list_servers():
            if server.name == lookup:
                return server
        raise OllamaServerNotFoundError(f'Ollama server "{lookup}" not found.')

    def add_server(self, *, name: str, url: str, replace: bool = False) -> OllamaServer:
        server_name = name.strip()
        if not server_name:
            raise OllamaServerError("Server name cannot be empty.")
        server_url = self._validate_url(url)

        servers = self.list_servers()
        existing_idx = None
        for idx, existing in enumerate(servers):
            if existing.name == server_name:
                existing_idx = idx
                break

        new_server = OllamaServer(name=server_name, url=server_url)
        if existing_idx is not None:
            if not replace:
                raise OllamaServerError(
                    f'Server "{server_name}" already exists. Use --replace to update it.'
                )
            servers[existing_idx] = new_server
        else:
            servers.append(new_server)

        serialized = [{"name": server.name, "url": server.url} for server in servers]
        self._write_data({"version": 1, "servers": serialized})
        return new_server

    def remove_server(self, name: str) -> OllamaServer:
        lookup = name.strip()
        servers = self.list_servers()
        for idx, server in enumerate(servers):
            if server.name == lookup:
                removed = servers.pop(idx)
                serialized = [{"name": item.name, "url": item.url} for item in servers]
                self._write_data({"version": 1, "servers": serialized})
                return removed
        raise OllamaServerNotFoundError(f'Ollama server "{lookup}" not found.')
