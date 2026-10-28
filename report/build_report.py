"""Render the curated report Markdown to PDF with ReportLab.

The coordinating agent must run the PDF skill's artifact-operation marker
before first authoring, then visually inspect rendered pages before delivery.
This script does not perform submission acceptance.
"""
import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Preformatted, KeepTogether, Image)


def inline(text):
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


def styles():
    sheet = getSampleStyleSheet()
    sheet.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=18,
                            leading=22, textColor=colors.HexColor('#16324f'), spaceAfter=12))
    sheet.add(ParagraphStyle('ReportH1', fontName='Helvetica-Bold', fontSize=12,
                            leading=15, textColor=colors.HexColor('#16324f'),
                            spaceBefore=10, spaceAfter=6, keepWithNext=True))
    sheet.add(ParagraphStyle('ReportH2', fontName='Helvetica-Bold', fontSize=10.5,
                            leading=13, spaceBefore=7, spaceAfter=4, keepWithNext=True))
    sheet.add(ParagraphStyle('ReportBody', fontName='Helvetica', fontSize=9.3,
                            leading=12.8, alignment=TA_LEFT, spaceAfter=5,
                            splitLongWords=True, allowWidows=0, allowOrphans=0))
    sheet.add(ParagraphStyle('ReportCell', parent=sheet['ReportBody'], fontSize=8,
                            leading=10.5, spaceAfter=0))
    sheet.add(ParagraphStyle('ReportCode', fontName='Courier', fontSize=7,
                            leading=9.3, spaceBefore=4, spaceAfter=8))
    return sheet


def table_rows(lines):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    return [row for row in rows if not all(re.fullmatch(r':?-+:?', cell) for cell in row)]


def parse_markdown(text, sheet, width, base_directory):
    lines = text.splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            image_path = (base_directory / image_match.group(2)).resolve()
            picture = Image(str(image_path))
            ratio = min(width / picture.imageWidth, 110 * mm / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * ratio
            picture.drawHeight = picture.imageHeight * ratio
            picture.hAlign = 'CENTER'
            picture.keepWithNext = True
            story.append(picture)
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            story.append(Preformatted('\n'.join(code), sheet['ReportCode'], maxLineLength=104))
            i += 1
            continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i])
                i += 1
            rows = table_rows(block)
            data = [[Paragraph(inline(cell), sheet['ReportCell']) for cell in row] for row in rows]
            ncols = len(rows[0])
            if ncols == 5:
                fractions = [0.34, 0.17, 0.17, 0.17, 0.15]
                widths = [width * f for f in fractions]
            elif ncols == 6:
                fractions = [0.24, 0.13, 0.17, 0.13, 0.18, 0.15]
                # Normalize explicit width ratios to the available frame.
                widths = [width * f / sum(fractions) for f in fractions]
            else:
                widths = [width / ncols] * ncols
            table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eff5')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fb')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#9aafbf')),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 8)])
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), sheet['ReportTitle']))
            i += 1
            continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), sheet['ReportH1']))
            i += 1
            continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), sheet['ReportH2']))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line or next_line.startswith(('#', '|', '```', '![')):
                break
            paragraph.append(next_line)
            i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), sheet['ReportBody']))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c6d2dc'))
    canvas.line(doc.leftMargin, 16 * mm, A4[0] - doc.rightMargin, 16 * mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526676'))
    canvas.drawString(doc.leftMargin, 11.5 * mm, 'DASE7506 Mini-project 2 | Reliable LLM Candidate Selection')
    canvas.drawRightString(A4[0] - doc.rightMargin, 11.5 * mm, str(doc.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', type=Path, default=Path(__file__).with_name('report.md'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('MP2_Report.pdf'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=17 * mm, bottomMargin=22 * mm,
                            title='Compact Evidence Fusion for Reliable LLM Candidate Selection',
                            author='DASE7506 MP2 submission')
    story = parse_markdown(args.markdown.read_text(encoding='utf-8-sig'), styles(),
                           doc.width, args.markdown.resolve().parent)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
