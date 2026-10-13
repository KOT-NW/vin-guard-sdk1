"""The core SDK remains importable without the LangChain extra."""

import subprocess
import sys


def test_core_import_does_not_require_langchain():
    script = '''
import importlib.abc
import sys

class WithoutLangChain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langchain_core" or fullname.startswith("langchain_core."):
            raise ModuleNotFoundError("LangChain intentionally unavailable")

sys.meta_path.insert(0, WithoutLangChain())
from ai_gateway import GatewayClient, AsyncGatewayClient, ControlPlaneClient
assert "langchain_core" not in sys.modules
try:
    import ai_gateway.langchain
except ImportError as exc:
    assert "vin-guard-sdk[langchain]" in str(exc)
else:
    raise AssertionError("Optional dependency was not enforced")
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
