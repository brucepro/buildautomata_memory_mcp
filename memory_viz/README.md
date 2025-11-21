# Memory Graph 3D Visualization

Interactive 3D visualization of the BuildAutomata memory graph using Python, Plotly, and FastAPI.

## Features

### Visual Design
- **3D Force-Directed Layout**: Nodes naturally cluster based on connections
- **Color Coding**: Distinct colors for each category (404 unique categories)
- **Node Sizing**: Weighted by importance (60%) + connection count (40%)
- **High-Access Glow**: Nodes with >20 accesses are highlighted
- **Dark Neural Theme**: Space-blue background (#0a0e27) with cyan accents

### Interactivity
- **Hover**: Preview memory content in tooltip
- **Click**: Open detailed view with full content + connected memories
- **Search**: Find memories by content, category, or tags
- **Filter by Category**: Select specific category or view all
- **Filter by Importance**: Slider from 0.0 to 1.0
- **Connection Threshold**: Slider to hide low-degree edges (reduces clutter)
- **Camera Presets**: Overview, Top View, Side View, Front View
- **Full 3D Navigation**: Orbit, pan, zoom with mouse

### Statistics Dashboard
- Total memories: 1,128
- Total connections: 9,768
- Average connections: 8.66 per memory
- Categories: 404
- Connection density: 0.0119

## Installation

```bash
cd memory_viz
pip install -r requirements.txt
```

## Usage

### Run Web Server

```bash
python app.py
```

Then open http://localhost:8686 in your browser.

### Generate Standalone HTML

```bash
python visualizer.py
```

Opens `memory_graph_test.html` - a 6.2MB standalone file with embedded data.

## Architecture

```
memory_viz/
├── database.py        # SQLite memory loader
├── graph_builder.py   # NetworkX graph construction
├── visualizer.py      # Plotly 3D visualization
├── app.py             # FastAPI web server
├── templates/
│   └── index.html     # Web UI
└── requirements.txt   # Python dependencies
```

### Data Flow

1. **database.py** reads `memoryv012.db` SQLite database
2. **graph_builder.py** constructs NetworkX graph from memories + connections
3. **visualizer.py** computes 3D force-directed layout and generates Plotly figure
4. **app.py** serves interactive web UI with filtering and search
5. **index.html** renders Plotly graph with dark neural theme + controls

## API Endpoints

- `GET /` - Main visualization page
- `GET /api/update_visualization` - Update graph with filters (category, importance, connection threshold)
- `GET /api/memory/{id}` - Get full memory details
- `GET /api/search?query={text}` - Search memories
- `GET /api/stats` - Get graph statistics

## Node Sizing Formula

```python
size = 5 + (importance * 0.6 + normalized_connections * 0.4) * 20
# Results in 5-25 pixel range
```

## Color Generation

Categories are assigned colors by distributing hues evenly around HSV color wheel:
- Hue: `i / total_categories` (0.0 to 1.0)
- Saturation: 0.7 (vibrant)
- Value: 0.85 (bright)

Converted to hex RGB for consistent appearance.

## Performance

- **Layout computation**: ~60-90 seconds for 1,128 nodes (NetworkX spring_layout, 50 iterations)
- **Rendering**: Smooth 60fps camera movement (Plotly WebGL)
- **File size**: 6.2MB standalone HTML (includes full dataset)
- **Memory usage**: ~200MB for graph construction + visualization

## Configuration

Default settings in `app.py`:
- Connection threshold: 1 (medium)
- Importance threshold: 0.0 (show all)
- High-access threshold: 20 accesses
- Camera eye: (1.5, 1.5, 1.5)
- Port: 8080

## Troubleshooting

### ModuleNotFoundError

```bash
pip install plotly networkx fastapi uvicorn jinja2 numpy scipy
```

### sqlite3.OperationalError: no such table

Database path is `../memoryv012.db` relative to `memory_viz/` directory. Adjust in `database.py` if needed:

```python
db = MemoryDatabase("path/to/memoryv012.db")
```

### Layout computation too slow

Reduce iterations in `visualizer.py`:

```python
pos = nx.spring_layout(graph, dim=3, iterations=30)  # Default is 50
```

## Technical Details

### Force-Directed Algorithm

Uses NetworkX's `spring_layout` with Fruchterman-Reingold algorithm:
- Nodes repel (like charged particles)
- Edges attract (like springs)
- Iteratively converges to equilibrium
- k-parameter: `2/sqrt(N)` for optimal spacing

### WebGL Rendering

Plotly uses Three.js under the hood:
- Nodes: `go.Scatter3d` with marker mode
- Edges: `go.Scatter3d` with line mode
- Camera: Trackball controls (orbit + zoom + pan)
- Background: Custom dark theme via layout config

## Credits

- **3D Force Graph Library Research**: vasturiano/3d-force-graph, TheBrain, InfraNodus
- **UX Research**: Duke Brain Portal, BRAIN Initiative, Neo4j graph visualization
- **Design Inspiration**: Neuroscience visualizations, personal knowledge graphs (Obsidian, Roam)

## License

Part of BuildAutomata memory system project.
