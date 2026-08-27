from __future__ import annotations

from localloop.policy import InteractiveApprovalPolicy


def test_interactive_policy_yes_no_auto_and_eof():
    output = []
    yes = InteractiveApprovalPolicy(input_fn=lambda _prompt: "yes", output_fn=output.append)
    assert yes.approve("write", "a.py", "diff") is True
    assert "diff" in output
    no = InteractiveApprovalPolicy(input_fn=lambda _prompt: "no", output_fn=lambda _text: None)
    assert no.approve("write", "a.py") is False
    auto = InteractiveApprovalPolicy(auto_approve=True, output_fn=output.append)
    assert auto.approve("command", "pytest") is True
    assert "[auto-approved]" in output

    def eof(_prompt):
        raise EOFError

    denied = InteractiveApprovalPolicy(input_fn=eof, output_fn=output.append)
    assert denied.approve("command", "pytest") is False
    assert "Denied." in output
