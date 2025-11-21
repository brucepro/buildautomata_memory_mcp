"""
FastAPI web application for memory graph visualization.
Serves interactive 3D visualization with controls.
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import plotly
import json
from typing import Optional

from database import MemoryDatabase
from graph_builder import MemoryGraph
from visualizer import MemoryVisualizer


app = FastAPI(title="Memory Graph Visualization")

# Templates
templates = Jinja2Templates(directory="templates")

# Global state
db = MemoryDatabase()
db.connect()
memories = db.load_all_memories()
memory_graph = MemoryGraph(memories)
db_stats = db.get_stats()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main visualization page."""
    # Get initial visualization
    viz = MemoryVisualizer(memory_graph)
    fig = viz.create_visualization(
        connection_threshold=1,  # Medium start as requested
        show_high_access_glow=True
    )

    # Convert to JSON for template
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Get graph stats
    graph_stats = memory_graph.get_stats()

    # Get categories
    categories = sorted(set(memory_graph.graph.nodes[n]['category'] for n in memory_graph.graph.nodes()))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "graph_json": graph_json,
            "stats": {
                **db_stats,
                **graph_stats
            },
            "categories": categories
        }
    )


@app.get("/api/update_visualization")
async def update_visualization(
    connection_threshold: int = 0,
    importance_threshold: float = 0.0,
    category: Optional[str] = None,
    show_high_access_glow: bool = True,
    importance_weight: float = 0.6,
    connection_weight: float = 0.4,
    layout_mode: str = 'force',
    color_mode: str = 'category',
    show_particles: bool = False,
    highlight_path: Optional[str] = None
):
    """
    Update visualization with filters.

    Query parameters:
    - connection_threshold: Minimum connections for edges to show
    - importance_threshold: Minimum importance for nodes to show
    - category: Filter by category (or null for all)
    - show_high_access_glow: Highlight high-access nodes
    - importance_weight: Weight for importance in sizing
    - connection_weight: Weight for connections in sizing
    - layout_mode: 'force', 'temporal', or 'cluster'
    - color_mode: 'category' or 'community'
    - show_particles: Show particle effects
    """
    # Apply filters
    filtered_graph = memory_graph

    if category and category != "all":
        filtered_graph = memory_graph.filter_by_category(category)

    if importance_threshold > 0:
        filtered_graph = filtered_graph.filter_by_importance(importance_threshold)

    # Parse highlight_path parameter
    path_list = None
    if highlight_path:
        path_list = [node_id.strip() for node_id in highlight_path.split(',') if node_id.strip()]

    # Create visualization
    viz = MemoryVisualizer(filtered_graph)
    fig = viz.create_visualization(
        connection_threshold=connection_threshold,
        show_high_access_glow=show_high_access_glow,
        importance_weight=importance_weight,
        connection_weight=connection_weight,
        layout_mode=layout_mode,
        color_mode=color_mode,
        show_particles=show_particles,
        highlight_path=path_list
    )

    # Return as JSON
    return JSONResponse(content=json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))


@app.get("/api/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Get full memory details by ID."""
    memory = db.get_memory_by_id(memory_id)

    if not memory:
        return JSONResponse(content={"error": "Memory not found"}, status_code=404)

    # Get connected memories
    connected = []
    if memory.related_memories:
        for related_id in memory.related_memories[:10]:  # Limit to 10
            related = db.get_memory_by_id(related_id)
            if related:
                connected.append({
                    "id": related.id,
                    "category": related.category,
                    "importance": related.importance,
                    "preview": related.preview
                })

    return JSONResponse(content={
        "id": memory.id,
        "content": memory.content,
        "category": memory.category,
        "importance": memory.importance,
        "tags": memory.tags,
        "created_at": memory.created_at,
        "access_count": memory.access_count,
        "connections": len(memory.related_memories),
        "connected_memories": connected
    })


@app.get("/api/search")
async def search_memories(query: str):
    """Search memories by content."""
    query_lower = query.lower()

    results = []
    for memory in memories:
        if (query_lower in memory.content.lower() or
            query_lower in memory.category.lower() or
            any(query_lower in tag.lower() for tag in memory.tags)):

            results.append({
                "id": memory.id,
                "category": memory.category,
                "importance": memory.importance,
                "preview": memory.preview,
                "created_at": memory.created_at
            })

    return JSONResponse(content={"results": results[:50]})  # Limit to 50 results


@app.get("/api/stats")
async def get_stats():
    """Get graph and database statistics."""
    graph_stats = memory_graph.get_stats()

    return JSONResponse(content={
        **db_stats,
        **graph_stats
    })


@app.get("/api/path/{source}/{target}")
async def find_path(source: str, target: str):
    """Find shortest path between two memories."""
    path = memory_graph.find_shortest_path(source, target)

    if not path:
        return JSONResponse(
            content={"error": "No path found between nodes"},
            status_code=404
        )

    return JSONResponse(content={
        "source": source,
        "target": target,
        "path": path,
        "length": len(path)
    })


@app.get("/api/export_png")
async def export_png():
    """Export current visualization as PNG (placeholder)."""
    # This would use plotly's static image export
    # Requires kaleido package
    return JSONResponse(content={
        "message": "PNG export requires kaleido package. Install with: pip install kaleido"
    })


if __name__ == "__main__":
    import uvicorn

    print("Starting Memory Graph Visualization server...")
    print("Open http://localhost:8080 in your browser")

    uvicorn.run(app, host="0.0.0.0", port=8686, log_level="info")
