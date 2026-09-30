from pathlib import Path
from types import SimpleNamespace

from tools import tts_tool_local


def test_neutts_uses_configured_sidecar_python(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, timeout):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stderr="OK: rendered")

    monkeypatch.setattr(tts_tool_local, "_run_helper", fake_run)
    monkeypatch.setattr(
        tts_tool_local,
        "_finalize_wav_output",
        lambda _wav, output: output,
    )

    output = str(tmp_path / "speech.mp3")
    result = tts_tool_local._generate_neutts(
        "hello",
        output,
        {"neutts": {"python": "/opt/neutts/bin/python"}},
    )

    assert result == output
    assert seen["cmd"][0] == "/opt/neutts/bin/python"
    assert seen["timeout"] == 120
