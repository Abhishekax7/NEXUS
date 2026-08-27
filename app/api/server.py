from app.api.app import (
    create_app,
)
from app.api.control_plane import (
    NexusControlPlane,
)
from app.core.runtime import (
    build_nexus_engine,
)


def build_production_app():
    """
    Build the complete production
    NEXUS FastAPI application.
    """

    engine = build_nexus_engine()

    control_plane = NexusControlPlane(
        engine
    )

    return create_app(
        control_plane
    )


app = build_production_app()
