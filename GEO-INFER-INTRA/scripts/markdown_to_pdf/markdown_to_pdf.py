#!/usr/bin/env python3
"""
Markdown to PDF Converter with Mermaid Support

This script converts Markdown files to PDF format with full support for Mermaid diagrams.
It automatically checks for and installs required dependencies.

Usage:
    python markdown_to_pdf.py input.md [--output-dir /path/to/output] [--config config.json]
    python markdown_to_pdf.py docs/*.md --output-dir ./pdfs/
    python markdown_to_pdf.py --help

Dependencies:
    - Node.js (automatically checked)
    - md-to-pdf npm package (automatically installed)

Author: GEO-INFER Team
License: MIT
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


class MarkdownToPDFConverter:
    """Markdown to PDF converter with Mermaid diagram support."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the converter (library-passive logging)."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    def check_node_js(self) -> bool:
        """Check if Node.js is installed and accessible."""
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, check=True
            )
            version = result.stdout.strip()
            self.logger.info(f"Node.js found: {version}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_npm(self) -> bool:
        """Check if npm is installed and accessible."""
        try:
            result = subprocess.run(
                ["npm", "--version"], capture_output=True, text=True, check=True
            )
            version = result.stdout.strip()
            self.logger.info(f"npm found: {version}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_md_to_pdf(self) -> bool:
        """Check if md-to-pdf is installed globally."""
        try:
            result = subprocess.run(
                ["md-to-pdf", "--version"], capture_output=True, text=True, check=True
            )
            version = result.stdout.strip()
            self.logger.info(f"md-to-pdf found: {version}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_md_to_pdf(self) -> bool:
        """Install md-to-pdf globally via npm."""
        try:
            self.logger.info("Installing md-to-pdf globally...")
            subprocess.run(
                ["npm", "install", "-g", "md-to-pdf"], check=True, capture_output=True
            )
            self.logger.info("md-to-pdf installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install md-to-pdf: {e}")
            return False

    def ensure_dependencies(self) -> bool:
        """Ensure all required dependencies are available."""
        self.logger.info("Checking dependencies...")

        # Check Node.js
        if not self.check_node_js():
            self.logger.error("Node.js is not installed. Please install Node.js first.")
            self.logger.info("Visit: https://nodejs.org/en/download/")
            return False

        # Check npm
        if not self.check_npm():
            self.logger.error("npm is not installed. Please install npm first.")
            return False

        # Check and install md-to-pdf
        if not self.check_md_to_pdf():
            self.logger.info("md-to-pdf not found. Installing...")
            if not self.install_md_to_pdf():
                return False

            # Verify installation
            if not self.check_md_to_pdf():
                self.logger.error("md-to-pdf installation verification failed")
                return False

        self.logger.info("All dependencies are ready!")
        return True

    def get_default_config(self) -> dict:
        """Get default configuration for PDF generation."""
        return {
            "stylesheet": [
                "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css"
            ],
            "css": """
                .markdown-body {
                    box-sizing: border-box;
                    min-width: 200px;
                    max-width: 980px;
                    margin: 0 auto;
                    padding: 45px;
                    font-size: 12px;
                    line-height: 1.6;
                }
                .page-break {
                    page-break-after: always;
                }
                .mermaid {
                    text-align: center;
                    margin: 20px 0;
                }
                @media print {
                    .markdown-body {
                        font-size: 11px;
                    }
                    .mermaid {
                        break-inside: avoid;
                    }
                }
            """,
            "body_class": ["markdown-body"],
            "highlight_style": "github",
            "pdf_options": {
                "format": "A4",
                "margin": "20mm",
                "printBackground": True,
                "preferCSSPageSize": True,
            },
            "launch_options": {"args": ["--no-sandbox", "--disable-setuid-sandbox"]},
        }

    def load_custom_config(self, config_path: Path) -> dict[str, Any]:
        """Load custom configuration from file."""
        try:
            with open(config_path) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    def create_config_file(self, config: dict, output_path: Path) -> Path:
        """Create a temporary config file for md-to-pdf."""
        config_file = output_path.parent / f".md-to-pdf-config-{output_path.stem}.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            return config_file
        except Exception as e:
            self.logger.error(f"Failed to create config file: {e}")
            raise

    def convert_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
        config: dict | None = None,
    ) -> bool:
        """Convert a single markdown file to PDF."""
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return False

        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".pdf")
        else:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use default config if none provided
        if config is None:
            config = self.get_default_config()

        # Set destination in config
        config["dest"] = str(output_path)

        # Create temporary config file
        config_file = None
        try:
            config_file = self.create_config_file(config, output_path)

            # Build command
            cmd = ["md-to-pdf", "--config-file", str(config_file), str(input_path)]

            self.logger.info(f"Converting {input_path} to {output_path}")

            # Execute conversion
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if output_path.exists():
                self.logger.info(f"Successfully created: {output_path}")
                return True
            else:
                self.logger.error(f"PDF was not created: {output_path}")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during conversion: {e}")
            return False
        finally:
            # Clean up temporary config file
            if config_file and config_file.exists():
                try:
                    config_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up config file: {e}")

    def convert_multiple(
        self,
        input_patterns: list[str],
        output_dir: Path | None = None,
        config: dict | None = None,
    ) -> dict[str, bool]:
        """Convert multiple markdown files to PDF."""
        results: dict[str, bool] = {}

        # Collect all matching files
        input_files = []
        for pattern in input_patterns:
            pattern_path = Path(pattern)
            if pattern_path.is_file():
                input_files.append(pattern_path)
            else:
                # Use glob to find matching files
                parent_dir = (
                    pattern_path.parent
                    if pattern_path.parent != Path(".")
                    else Path.cwd()
                )
                pattern_name = pattern_path.name
                matching_files = list(parent_dir.glob(pattern_name))
                input_files.extend(
                    [
                        f
                        for f in matching_files
                        if f.suffix.lower() in [".md", ".markdown"]
                    ]
                )

        if not input_files:
            self.logger.warning("No markdown files found matching the patterns")
            return results

        self.logger.info(f"Found {len(input_files)} markdown files to convert")

        # Convert each file
        for input_file in input_files:
            if output_dir:
                output_path = output_dir / f"{input_file.stem}.pdf"
            else:
                output_path = input_file.with_suffix(".pdf")

            success = self.convert_file(input_file, output_path, config)
            results[str(input_file)] = success

        return results


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert Markdown files to PDF with Mermaid diagram support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.md
  %(prog)s document.md --output-dir ./pdfs/
  %(prog)s docs/*.md --output-dir ./output/
  %(prog)s document.md --config custom-config.json
  %(prog)s document.md --log-level DEBUG
        """,
    )

    parser.add_argument(
        "input_files", nargs="*", help="Input markdown file(s) or glob patterns"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for PDF files (default: same as input file)",
    )

    parser.add_argument(
        "--config", type=Path, help="Custom configuration file (JSON format)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Only check dependencies and exit"
    )

    args = parser.parse_args()
    # Initialize converter
    converter = MarkdownToPDFConverter(log_level=args.log_level)

    # Check dependencies
    if not converter.ensure_dependencies():
        sys.exit(1)

    if args.check_deps:
        converter.logger.info("Dependency check completed successfully")
        sys.exit(0)

    # Check if input files are provided
    if not args.input_files:
        converter.logger.error(
            "No input files provided. Use --help for usage information."
        )
        sys.exit(1)

    # Load configuration
    config = converter.get_default_config()
    if args.config:
        custom_config = converter.load_custom_config(args.config)
        config.update(custom_config)

    # Perform conversions
    results = converter.convert_multiple(
        input_patterns=args.input_files, output_dir=args.output_dir, config=config
    )

    # Report results
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    converter.logger.info(f"Conversion completed: {successful}/{total} successful")

    if successful < total:
        converter.logger.warning("Some conversions failed:")
        for file_path, success in results.items():
            if not success:
                converter.logger.warning(f"  Failed: {file_path}")
        sys.exit(1)
    else:
        converter.logger.info("All conversions completed successfully!")


if __name__ == "__main__":
    # CLI entry point: configure root logging here, not in the library class.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
