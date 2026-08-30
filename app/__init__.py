try:  # ADK agent (agent service). The UI image imports only app.cloud_tools
    from .agent import app, root_agent  # noqa: F401
    __all__ = ["app", "root_agent"]
except ImportError:  # google-adk not installed (e.g. toposcout-ui container)
    __all__: list[str] = []
