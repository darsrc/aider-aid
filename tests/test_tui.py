import aider_aid.tui as tui


def test_select_index_hides_question_mark(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeQuestion:
        def ask(self):  # noqa: ANN001
            return 0

    def fake_select(message, choices, **kwargs):  # noqa: ANN001
        captured["message"] = message
        captured["kwargs"] = kwargs
        return _FakeQuestion()

    monkeypatch.setattr(tui.questionary, "select", fake_select)
    result = tui.select_index("Choose:", ["A"])
    assert result == 0
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["qmark"] == ""


def test_ask_text_hides_question_mark(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeQuestion:
        def ask(self):  # noqa: ANN001
            return "abc"

    def fake_text(message, **kwargs):  # noqa: ANN001
        captured["message"] = message
        captured["kwargs"] = kwargs
        return _FakeQuestion()

    monkeypatch.setattr(tui.questionary, "text", fake_text)
    result = tui.ask_text("Name")
    assert result == "abc"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["qmark"] == ""


def test_ask_confirm_hides_question_mark(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeQuestion:
        def ask(self):  # noqa: ANN001
            return True

    def fake_confirm(message, **kwargs):  # noqa: ANN001
        captured["message"] = message
        captured["kwargs"] = kwargs
        return _FakeQuestion()

    monkeypatch.setattr(tui.questionary, "confirm", fake_confirm)
    result = tui.ask_confirm("Proceed?")
    assert result is True
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["qmark"] == ""
