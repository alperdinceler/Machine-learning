from google import genai
from typing import List, Dict, Optional
import os
import re
import html
import logging
from xhtml2pdf import pisa
from summarizer import call_gemini_with_backoff
import config

def generate_report(client: Optional[genai.Client], query: str, sources: List[Dict[str, str]], model: str = config.DEFAULT_MODEL) -> Optional[str]:
    """
    Generates a structured research report with citations based on source summaries.
    Returns clean markdown report or None if generation fails.
    """
    if not sources:
        return None

    sources_context = ""
    for idx, src in enumerate(sources, 1):
        sources_context += f"Source [{idx}]:\nTitle: {src['title']}\nURL: {src['url']}\nSummary: {src['summary']}\n\n"

    prompt = f"""
You are an elite Research Assistant. Generate a professional, highly detailed, and objective research report answering the query:
"{query}"

Use the following sources to draft your report. You MUST cite the sources in-text using markdown numbers (e.g., [1], [2], etc.) when referencing facts from them.

Sources:
{sources_context}

Guidelines:
1. Structure the report with a Title, Executive Summary, Detailed Analysis (split into relevant thematic subheadings), and a Conclusion.
2. Write the report in Turkish (since the request is in Turkish) unless the user asks otherwise.
3. Ensure every major claim has an in-text citation pointing to the correct Source number.
4. Do not invent facts. Only use information provided in the sources.
5. At the end of the report, add a references section titled exactly "Kaynakça" or "Kaynakça (References)" mapping each source number to its Title and URL.
"""

    try:
        res = call_gemini_with_backoff(client, model, prompt, delay_before=0.5)
        return res if res and res.strip() else None
    except Exception as e:
        logging.error(f"Failed to generate report: {e}")
        return None

def markdown_to_html(md_text: str) -> str:
    """
    Converts standard Markdown report to HTML supporting headers, bold, italics, links, and ordered/unordered lists.
    Safely handles HTML escaping and Turkish characters.
    """
    if not md_text:
        return ""

    html_lines = []
    lines = md_text.split('\n')
    current_list_type = None  # 'ul' or 'ol'

    def close_list_if_open():
        nonlocal current_list_type
        if current_list_type:
            html_lines.append(f"</{current_list_type}>")
            current_list_type = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list_if_open()
            continue

        # Headers
        if stripped.startswith("### "):
            close_list_if_open()
            html_lines.append(f"<h3>{_format_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_list_if_open()
            html_lines.append(f"<h2>{_format_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_list_if_open()
            html_lines.append(f"<h1>{_format_inline(stripped[2:])}</h1>")
        # Unordered Lists
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if current_list_type != 'ul':
                close_list_if_open()
                html_lines.append("<ul>")
                current_list_type = 'ul'
            html_lines.append(f"<li>{_format_inline(stripped[2:])}</li>")
        # Ordered Lists
        elif re.match(r'^\d+\.\s+', stripped):
            if current_list_type != 'ol':
                close_list_if_open()
                html_lines.append("<ol>")
                current_list_type = 'ol'
            content = re.sub(r'^\d+\.\s+', '', stripped)
            html_lines.append(f"<li>{_format_inline(content)}</li>")
        else:
            close_list_if_open()
            html_lines.append(f"<p>{_format_inline(stripped)}</p>")

    close_list_if_open()
    return "\n".join(html_lines)

def _format_inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    return text

def save_pdf(report_md: str, output_pdf_path: str) -> bool:
    """
    Converts a Markdown report into a styled PDF.
    """
    if not report_md:
        return False

    html_body = markdown_to_html(report_md)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
        <style>
            @page {{
                size: a4;
                margin: 2cm;
                @frame footer_frame {{
                    -pdf-frame-content: footer_content;
                    bottom: 1cm;
                    left: 2cm;
                    right: 2cm;
                    height: 1cm;
                }}
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                color: #333333;
                line-height: 1.5;
                font-size: 10pt;
            }}
            h1 {{
                font-size: 20pt;
                color: #1A365D;
                text-align: center;
                margin-bottom: 20px;
            }}
            h2 {{
                font-size: 14pt;
                color: #2B6CB0;
                border-bottom: 1px solid #E2E8F0;
                padding-bottom: 5px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            h3 {{
                font-size: 11pt;
                color: #2D3748;
                margin-top: 15px;
                margin-bottom: 5px;
            }}
            p {{
                margin-bottom: 10px;
                text-align: justify;
            }}
            ul, ol {{
                margin-bottom: 10px;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            .footer {{
                text-align: center;
                font-size: 8pt;
                color: #718096;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            {html_body}
        </div>
        <div id="footer_content" class="footer">
            Türkçe Araştırma Asistanı Raporu - Sayfa <pdf:pagenumber>
        </div>
    </body>
    </html>
    """

    try:
        with open(output_pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_template, dest=pdf_file, encoding="utf-8")
        return not pisa_status.err
    except Exception as e:
        logging.error(f"[Warning] Failed to generate PDF: {e}")
        return False
