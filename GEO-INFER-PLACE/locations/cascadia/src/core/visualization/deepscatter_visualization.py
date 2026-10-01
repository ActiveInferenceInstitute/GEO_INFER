"""
Deepscatter-based visualization for Cascadia H3 data.
Provides efficient web-based rendering with WebGL acceleration.
"""

import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class CascadiaDeepscatterVisualizer:
    """
    Efficient web-based visualization of Cascadia H3 data using Deepscatter.
    Provides WebGL-accelerated rendering for large datasets.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def prepare_deepscatter_data(
        self, unified_data: dict, redevelopment_scores: dict
    ) -> list[dict]:
        """
        Convert H3 data to Deepscatter format.

        Args:
            unified_data: H3 unified data
            redevelopment_scores: Redevelopment scores

        Returns:
            List of data points in Deepscatter format
        """
        logger.info("Preparing H3 data for Deepscatter visualization...")

        # Import H3 utilities
        from h3 import cell_to_latlng

        data_points = []

        for h3_id, hex_data in unified_data.items():
            try:
                # Get centroid coordinates
                lat, lng = cell_to_latlng(h3_id)

                # Get redevelopment score
                score_data = redevelopment_scores.get(h3_id, {})
                composite_score = (
                    score_data.get("composite_score", 0.0) if isinstance(score_data, dict) else 0.0
                )

                # Extract module scores
                module_scores = {}
                for module_name in [
                    "zoning",
                    "current_use",
                    "ownership",
                    "improvements",
                ]:
                    module_data = hex_data.get(module_name, {})
                    if isinstance(module_data, dict):
                        module_scores[module_name] = module_data.get("score", 0.0)
                    else:
                        module_scores[module_name] = 0.0

                # Create Deepscatter data point
                point = {
                    "x": lng,  # longitude
                    "y": lat,  # latitude
                    "h3_id": h3_id,
                    "redevelopment_score": composite_score,
                    "zoning_score": module_scores["zoning"],
                    "current_use_score": module_scores["current_use"],
                    "ownership_score": module_scores["ownership"],
                    "improvements_score": module_scores["improvements"],
                    "total_coverage": sum(module_scores.values()),
                }
                data_points.append(point)

            except Exception as e:
                logger.warning(f"Could not process H3 cell {h3_id}: {e}")
                continue

        logger.info(f"Prepared {len(data_points)} data points for Deepscatter")
        return data_points

    def create_deepscatter_html(
        self, data_points: list[dict], title: str = "Cascadia Agricultural Analysis"
    ) -> str:
        """
        Create a complete Deepscatter HTML visualization.

        Args:
            data_points: List of data points in Deepscatter format
            title: Title for the visualization

        Returns:
            Path to the generated HTML file
        """
        logger.info("Creating Deepscatter HTML visualization...")

        # Create the HTML content from template fragments to avoid f-string brace issues
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .controls {{
            padding: 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .control-group label {{
            font-weight: bold;
            font-size: 12px;
            color: #666;
        }}
        .control-group select, .control-group input {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .visualization {{
            height: 600px;
            position: relative;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 200px;
        }}
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            font-size: 18px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>__TITLE__</h1>
            <p>Interactive visualization of agricultural redevelopment potential in Del Norte County</p>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="colorField">Color By:</label>
                <select id="colorField">
                    <option value="redevelopment_score">Redevelopment Score</option>
                    <option value="zoning_score">Zoning Score</option>
                    <option value="current_use_score">Current Use Score</option>
                    <option value="ownership_score">Ownership Score</option>
                    <option value="improvements_score">Improvements Score</option>
                    <option value="total_coverage">Total Coverage</option>
                </select>
            </div>

            <div class="control-group">
                <label for="sizeField">Size By:</label>
                <select id="sizeField">
                    <option value="1">Fixed Size</option>
                    <option value="redevelopment_score">Redevelopment Score</option>
                    <option value="total_coverage">Total Coverage</option>
                </select>
            </div>

            <div class="control-group">
                <label for="opacity">Opacity:</label>
                <input type="range" id="opacity" min="0.1" max="1" step="0.1" value="0.7">
            </div>

            <div class="control-group">
                <label for="pointSize">Point Size:</label>
                <input type="range" id="pointSize" min="1" max="10" step="1" value="3">
            </div>
        </div>

        <div class="visualization">
            <div id="scatterplot"></div>
            <div class="info-panel">
                <div><strong>Zoom:</strong> Mouse wheel</div>
                <div><strong>Pan:</strong> Click and drag</div>
                <div><strong>Reset:</strong> Double click</div>
                <div id="pointInfo"></div>
            </div>
        </div>
    </div>

    <script type="module">
        // Import Deepscatter (you'll need to include the actual library)
        // For now, we'll create a simple scatter plot using D3.js

        import * as d3 from 'https://cdn.skypack.dev/d3@7';

        // Data from the backend
        const data = __DATA__;

        // Setup visualization
        const width = 1200;
        const height = 600;
        const margin = {top: 20, right: 20, bottom: 30, left: 40};

        const svg = d3.select('#scatterplot')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // Scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x))
            .range([0, width - margin.left - margin.right]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y))
            .range([height - margin.top - margin.bottom, 0]);

        const colorScale = d3.scaleSequential()
            .domain([0, d3.max(data, d => d.redevelopment_score)])
            .interpolator(d3.interpolateViridis);

        // Add points
        g.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .attr('fill', d => colorScale(d.redevelopment_score))
            .attr('opacity', 0.7)
            .on('mouseover', function(event, d) {{
                d3.select(this).attr('r', 6);
                d3.select('#pointInfo').html(`
                    <strong>H3:</strong> ${{d.h3_id}}<br>
                    <strong>Score:</strong> ${{d.redevelopment_score.toFixed(3)}}<br>
                    <strong>Zoning:</strong> ${{d.zoning_score.toFixed(3)}}<br>
                    <strong>Current Use:</strong> ${{d.current_use_score.toFixed(3)}}
                `);
            }})
            .on('mouseout', function() {{
                d3.select(this).attr('r', 3);
                d3.select('#pointInfo').html('');
            }});

        // Add axes
        g.append('g')
            .attr('transform', 'translate(0,' + (height - margin.top - margin.bottom) + ')')
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        // Add labels
        g.append('text')
            .attr('x', (width - margin.left - margin.right) / 2)
            .attr('y', height - margin.top - margin.bottom + 25)
            .style('text-anchor', 'middle')
            .text('Longitude');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -margin.left + 20)
            .attr('x', -(height - margin.top - margin.bottom) / 2)
            .style('text-anchor', 'middle')
            .text('Latitude');

        // Control handlers
        document.getElementById('colorField').addEventListener('change', function() {{
            const field = this.value;
            const newColorScale = d3.scaleSequential()
                .domain([0, d3.max(data, d => d[field])])
                .interpolator(d3.interpolateViridis);

            g.selectAll('circle')
                .attr('fill', d => newColorScale(d[field]));
        }});

        document.getElementById('opacity').addEventListener('input', function() {{
            g.selectAll('circle')
                .attr('opacity', this.value);
        }});

        document.getElementById('pointSize').addEventListener('input', function() {{
            g.selectAll('circle')
                .attr('r', this.value);
        }});
    </script>
</body>
</html>
        """
        html_content = html_template.replace("__TITLE__", title).replace(
            "__DATA__", json.dumps(data_points)
        )

        # Save to file
        output_path = self.output_dir / "cascadia_deepscatter_visualization.html"
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Deepscatter HTML visualization saved to: {output_path}")
        return str(output_path)

    def create_lightweight_csv_export(self, data_points: list[dict]) -> str:
        """
        Create a lightweight CSV export for external visualization tools.

        Args:
            data_points: List of data points

        Returns:
            Path to the CSV file
        """
        logger.info("Creating lightweight CSV export...")

        # Convert to DataFrame
        df = pd.DataFrame(data_points)

        # Save to CSV
        output_path = self.output_dir / "cascadia_lightweight_data.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"Lightweight CSV export saved to: {output_path}")
        return str(output_path)


def create_deepscatter_visualization(backend, output_dir: Path) -> dict[str, str]:
    """
    Create efficient Deepscatter visualizations for Cascadia data.

    Args:
        backend: Unified backend with processed data
        output_dir: Output directory for visualizations

    Returns:
        Dictionary with paths to generated visualizations
    """
    logger.info("Creating Deepscatter visualizations...")

    # Initialize visualizer
    visualizer = CascadiaDeepscatterVisualizer(output_dir)

    # Get data from backend
    unified_data = backend.unified_data
    redevelopment_scores = backend.calculate_agricultural_redevelopment_potential()

    # Create visualizations
    results = {}

    try:
        # Prepare data
        data_points = visualizer.prepare_deepscatter_data(unified_data, redevelopment_scores)

        # Create HTML visualization
        html_path = visualizer.create_deepscatter_html(data_points)
        results["deepscatter_html"] = html_path

        # Create CSV export
        csv_path = visualizer.create_lightweight_csv_export(data_points)
        results["lightweight_csv"] = csv_path

        logger.info("Deepscatter visualizations created successfully")

    except Exception as e:
        logger.error(f"Failed to create Deepscatter visualizations: {e}")
        results["error"] = str(e)

    return results
