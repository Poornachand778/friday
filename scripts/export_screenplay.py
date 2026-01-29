#!/usr/bin/env python3
"""
Screenplay Export System for Friday AI
=======================================

Exports screenplay data from database to properly formatted:
- PDF with gray box scene headings (Celtx style)
- Fountain format
- HTML for email

Usage:
    python scripts/export_screenplay.py --project aa-janta-naduma --format pdf --output script.pdf
    python scripts/export_screenplay.py --project gusagusalu-script --format fountain --output script.fountain
    python scripts/export_screenplay.py --project can-we-not --format html --output script.html
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "vectoruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "friday")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_session():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


# Import schema models
from db.screenplay_schema import (
    ScreenplayProject,
    ScreenplayScene,
    SceneElement,
    DialogueLine,
    ExportConfig,
)


class ScreenplayExporter:
    """Export screenplay from database to various formats"""

    def __init__(self, project_slug: str, session):
        self.session = session
        self.project = (
            session.query(ScreenplayProject).filter_by(slug=project_slug).first()
        )
        if not self.project:
            raise ValueError(f"Project not found: {project_slug}")

        # Get export config
        self.config = (
            session.query(ExportConfig).filter_by(name="celtx_default").first()
        )
        if not self.config:
            # Use default values
            self.config = ExportConfig(
                font_family="Courier Prime",
                font_size=12,
                page_width=8.5,
                page_height=11.0,
                margin_top=1.0,
                margin_bottom=1.0,
                margin_left=1.5,
                margin_right=1.0,
                scene_heading_bg_color="#CCCCCC",
                scene_heading_bold=True,
                character_name_caps=True,
                parenthetical_italics=False,
                show_translations=True,
                translation_in_parentheses=True,
            )

    def export_fountain(self) -> str:
        """Export to Fountain format"""
        lines = []

        # Title page
        lines.append(f"Title: {self.project.title}")
        if self.project.author:
            lines.append(f"Author: {self.project.author}")
        if self.project.draft_date:
            lines.append(f"Draft date: {self.project.draft_date}")
        lines.append("")
        lines.append("")

        # Scenes
        for scene in self.project.scenes:
            # Scene heading
            heading = self._format_scene_heading(scene)
            lines.append(heading)
            lines.append("")

            # Elements
            for element in scene.elements:
                if element.element_type == "action":
                    action_text = element.content.get("text", "")
                    lines.append(action_text)
                    lines.append("")

                elif element.element_type == "dialogue":
                    char_name = element.content.get("character", "UNKNOWN")
                    paren = element.content.get("parenthetical")

                    lines.append(char_name.upper())
                    if paren:
                        lines.append(f"({paren})")

                    for line_data in element.content.get("lines", []):
                        text = line_data.get("text", "")
                        translation = line_data.get("translation")

                        if translation and self.config.show_translations:
                            if self.config.translation_in_parentheses:
                                lines.append(f"{text}")
                                lines.append(f"({translation})")
                            else:
                                lines.append(f"{text} -- {translation}")
                        else:
                            lines.append(text)

                    lines.append("")

                elif element.element_type == "transition":
                    trans_text = element.content.get("text", "CUT TO:")
                    lines.append(f"> {trans_text}")
                    lines.append("")

            lines.append("")

        return "\n".join(lines)

    def export_html(self) -> str:
        """Export to HTML format (for email)"""
        html = []

        # HTML header
        html.append(
            """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: 'Courier Prime', 'Courier New', monospace;
            font-size: 12pt;
            max-width: 8.5in;
            margin: 1in auto;
            padding: 0 1.5in 0 1in;
            line-height: 1.5;
        }
        .title-page {
            text-align: center;
            margin-top: 3in;
        }
        .title {
            font-size: 18pt;
            font-weight: bold;
            text-transform: uppercase;
        }
        .scene-heading {
            background-color: #CCCCCC;
            font-weight: bold;
            text-transform: uppercase;
            padding: 4px 8px;
            margin: 24px 0 12px 0;
        }
        .action {
            margin: 12px 0;
        }
        .character {
            text-transform: uppercase;
            margin-left: 2.5in;
            margin-top: 12px;
        }
        .parenthetical {
            margin-left: 2in;
            font-style: italic;
        }
        .dialogue {
            margin-left: 1.5in;
            margin-right: 1.5in;
        }
        .translation {
            color: #666;
            font-style: italic;
        }
        .transition {
            text-align: right;
            text-transform: uppercase;
            margin: 12px 0;
        }
    </style>
</head>
<body>
"""
        )

        # Title page
        html.append(
            f"""
<div class="title-page">
    <div class="title">{self.project.title}</div>
    <br><br>
    <div>Written by</div>
    <div>{self.project.author or 'Friday AI'}</div>
</div>
<div style="page-break-after: always;"></div>
"""
        )

        # Scenes
        for scene in self.project.scenes:
            heading = self._format_scene_heading(scene)
            html.append(f'<div class="scene-heading">{heading}</div>')

            for element in scene.elements:
                if element.element_type == "action":
                    text = element.content.get("text", "").replace("\n", "<br>")
                    html.append(f'<div class="action">{text}</div>')

                elif element.element_type == "dialogue":
                    char_name = element.content.get("character", "UNKNOWN")
                    paren = element.content.get("parenthetical")

                    html.append(f'<div class="character">{char_name.upper()}</div>')
                    if paren:
                        html.append(f'<div class="parenthetical">({paren})</div>')

                    for line_data in element.content.get("lines", []):
                        text = line_data.get("text", "")
                        translation = line_data.get("translation")

                        html.append(f'<div class="dialogue">{text}</div>')
                        if translation and self.config.show_translations:
                            html.append(
                                f'<div class="dialogue translation">({translation})</div>'
                            )

                elif element.element_type == "transition":
                    text = element.content.get("text", "CUT TO:")
                    html.append(f'<div class="transition">{text}</div>')

        html.append("</body></html>")
        return "\n".join(html)

    def export_pdf(self, output_path: str) -> bool:
        """Export to PDF with gray box scene headings"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        except ImportError:
            print("ERROR: reportlab not installed. Run: pip install reportlab")
            return False

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=self.config.margin_top * inch,
            bottomMargin=self.config.margin_bottom * inch,
            leftMargin=self.config.margin_left * inch,
            rightMargin=self.config.margin_right * inch,
        )

        # Styles
        styles = getSampleStyleSheet()

        # Custom styles for screenplay
        styles.add(
            ParagraphStyle(
                name="SceneHeading",
                fontName="Courier",
                fontSize=12,
                leading=14,
                textColor=HexColor("#000000"),
                backColor=HexColor(self.config.scene_heading_bg_color),
                spaceBefore=24,
                spaceAfter=12,
                leftIndent=0,
                rightIndent=0,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Action",
                fontName="Courier",
                fontSize=12,
                leading=14,
                spaceBefore=6,
                spaceAfter=6,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Character",
                fontName="Courier",
                fontSize=12,
                leading=14,
                spaceBefore=12,
                leftIndent=2 * inch,
                alignment=TA_LEFT,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Parenthetical",
                fontName="Courier",
                fontSize=12,
                leading=14,
                leftIndent=1.5 * inch,
                rightIndent=1.5 * inch,
                fontStyle="italic",
            )
        )

        styles.add(
            ParagraphStyle(
                name="Dialogue",
                fontName="Courier",
                fontSize=12,
                leading=14,
                leftIndent=1 * inch,
                rightIndent=1.5 * inch,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Translation",
                fontName="Courier",
                fontSize=10,
                leading=12,
                leftIndent=1 * inch,
                rightIndent=1.5 * inch,
                textColor=HexColor("#666666"),
            )
        )

        styles.add(
            ParagraphStyle(
                name="Transition",
                fontName="Courier",
                fontSize=12,
                leading=14,
                alignment=TA_RIGHT,
                spaceBefore=12,
                spaceAfter=12,
            )
        )

        # Build story
        story = []

        # Title page
        story.append(Spacer(1, 3 * inch))
        story.append(
            Paragraph(
                f"<b>{self.project.title.upper()}</b>",
                ParagraphStyle(
                    name="Title", fontName="Courier", fontSize=18, alignment=TA_CENTER
                ),
            )
        )
        story.append(Spacer(1, 0.5 * inch))
        story.append(
            Paragraph(
                "Written by",
                ParagraphStyle(
                    name="By", fontName="Courier", fontSize=12, alignment=TA_CENTER
                ),
            )
        )
        story.append(
            Paragraph(
                self.project.author or "Friday AI",
                ParagraphStyle(
                    name="Author", fontName="Courier", fontSize=12, alignment=TA_CENTER
                ),
            )
        )

        # Page break
        from reportlab.platypus import PageBreak

        story.append(PageBreak())

        # Scenes
        for scene in self.project.scenes:
            heading = self._format_scene_heading(scene)
            story.append(Paragraph(f"<b>{heading}</b>", styles["SceneHeading"]))

            for element in scene.elements:
                if element.element_type == "action":
                    text = element.content.get("text", "")
                    # Escape any HTML-like content
                    text = (
                        text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    text = text.replace("\n", "<br/>")
                    story.append(Paragraph(text, styles["Action"]))

                elif element.element_type == "dialogue":
                    char_name = element.content.get("character", "UNKNOWN")
                    paren = element.content.get("parenthetical")

                    story.append(Paragraph(char_name.upper(), styles["Character"]))

                    if paren:
                        story.append(Paragraph(f"({paren})", styles["Parenthetical"]))

                    for line_data in element.content.get("lines", []):
                        text = line_data.get("text", "")
                        translation = line_data.get("translation")

                        # Escape HTML
                        text = (
                            text.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        story.append(Paragraph(text, styles["Dialogue"]))

                        if translation and self.config.show_translations:
                            translation = (
                                translation.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            story.append(
                                Paragraph(f"({translation})", styles["Translation"])
                            )

                elif element.element_type == "transition":
                    text = element.content.get("text", "CUT TO:")
                    story.append(Paragraph(text.upper(), styles["Transition"]))

        # Build PDF
        doc.build(story)
        return True

    def _format_scene_heading(self, scene: ScreenplayScene) -> str:
        """Format scene heading string"""
        parts = [f"{scene.int_ext}."]

        if scene.location:
            parts.append(scene.location.upper())
        if scene.sub_location:
            parts.append(f"- {scene.sub_location.upper()}")
        if scene.time_of_day:
            parts.append(f"- {scene.time_of_day.upper()}")

        return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Export screenplay from database")
    parser.add_argument("--project", "-p", required=True, help="Project slug")
    parser.add_argument(
        "--format",
        "-f",
        choices=["pdf", "fountain", "html"],
        default="fountain",
        help="Export format (default: fountain)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--list", action="store_true", help="List all projects")
    args = parser.parse_args()

    session = get_session()

    try:
        if args.list:
            projects = session.query(ScreenplayProject).all()
            print("\nAvailable projects:")
            for p in projects:
                print(f"  - {p.slug} ({p.title})")
            return

        exporter = ScreenplayExporter(args.project, session)

        if args.format == "fountain":
            content = exporter.export_fountain()
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Exported to Fountain: {args.output}")

        elif args.format == "html":
            content = exporter.export_html()
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Exported to HTML: {args.output}")

        elif args.format == "pdf":
            success = exporter.export_pdf(args.output)
            if success:
                print(f"Exported to PDF: {args.output}")
            else:
                print("PDF export failed")
                return 1

    finally:
        session.close()


if __name__ == "__main__":
    main()
