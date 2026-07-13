import os
import sys
import argparse
import re
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

import config
from searcher import search_web
from scraper import scrape_page, scrape_urls
from summarizer import summarize_text
from reporter import generate_report, save_pdf

console = Console()

def slugify(text: str) -> str:
    """
    Slugify function for Turkish and English characters.
    Defaults to 'arastirma-raporu' if the query contains only punctuation/symbols.
    """
    text = text.lower()
    turkish_map = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'â': 'a', 'î': 'i', 'û': 'u'
    }
    for char, replacement in turkish_map.items():
        text = text.replace(char, replacement)
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text if text else "arastirma-raporu"

def check_playwright_installed():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                p.chromium.launch(headless=True)
            except Exception:
                console.print("[yellow]Playwright chromium browser not found. Attempting installation...[/yellow]")
                import subprocess
                subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        console.print(f"[yellow]Warning: Playwright setup check failed: {e}[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Turkish Research Assistant (Araştırma Asistanı) CLI")
    parser.add_argument("--query", type=str, help="Search query or topic to research")
    parser.add_argument("--results", type=int, default=config.MAX_SEARCH_RESULTS, help="Number of search results to fetch")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL, help="Gemini model to use")
    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]Araştırma Asistanı (Search & Summarize Pipeline)[/bold blue]", border_style="blue"))

    query = args.query
    if not query:
        if not sys.stdin.isatty():
            console.print("[red]Hata: Non-interactive ortamda --query parametresi zorunludur.[/red]")
            sys.exit(1)
        query = console.input("[bold yellow]Lütfen araştırmak istediğiniz konuyu girin: [/bold yellow]").strip()
        if not query:
            console.print("[red]Hata: Boş bir arama sorgusu girilemez.[/red]")
            sys.exit(1)

    api_key = config.GEMINI_API_KEY
    client = None
    if not api_key:
        if config.USE_LOCAL_FALLBACK:
            console.print("[yellow]Uyarı: GEMINI_API_KEY bulunamadı. Yerel model (Ollama) yedek olarak kullanılacak.[/yellow]")
        else:
            console.print("[yellow]Uyarı: GEMINI_API_KEY ortam değişkeni bulunamadı.[/yellow]")
            if not sys.stdin.isatty():
                console.print("[red]Hata: Non-interactive ortamda GEMINI_API_KEY bulunmalıdır.[/red]")
                sys.exit(1)
            api_key = console.input("[yellow]Lütfen Gemini API Anahtarınızı girin: [/yellow]").strip()
            if not api_key:
                console.print("[red]Hata: Gemini API anahtarı olmadan bu araç çalıştırılamaz.[/red]")
                sys.exit(1)
            os.environ["GEMINI_API_KEY"] = api_key

    if api_key:
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            if config.USE_LOCAL_FALLBACK:
                console.print(f"[yellow]Uyarı: Gemini Client başlatılamadı ({e}). Yerel model (Ollama) kullanılacak.[/yellow]")
                client = None
            else:
                console.print(f"[red]Gemini Client başlatılamadı: {e}[/red]")
                sys.exit(1)

    check_playwright_installed()

    # 1. Search Web
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task1 = progress.add_task(description=f"'{query}' için web aranıyor (Çok dilli arama)...", total=None)
        search_results = search_web(client, query, args.model, max_results=args.results)
        progress.update(task1, completed=True)

    if not search_results:
        console.print("[red]Arama sonucu bulunamadı. Lütfen başka bir sorgu deneyin.[/red]")
        sys.exit(1)

    console.print(f"[green]Bulunan kaynak sayısı (Tekilleştirilmiş): {len(search_results)}[/green]")
    for idx, r in enumerate(search_results, 1):
        console.print(f"  [bold][{idx}][/bold] {r['title']} - [dim]{r['url']}[/dim]")

    # 2. Scrape & Summarize (using single shared browser instance for efficiency)
    urls = [result["url"] for result in search_results]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        scrape_task = progress.add_task(description="Web sayfaları kazınıyor (Tek browser oturumu)...", total=None)
        scraped_contents = scrape_urls(urls, timeout_seconds=config.REQUEST_TIMEOUT)
        progress.update(scrape_task, completed=True)

    sources = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for idx, result in enumerate(search_results, 1):
            url = result["url"]
            title = result["title"]
            text = scraped_contents.get(url)

            task_id = progress.add_task(description=f"[{idx}/{len(search_results)}] {title[:40]}... özetleniyor...", total=None)
            if not text:
                progress.update(task_id, description=f"[red]Kazınamadı:[/red] {title[:40]}...", completed=True)
                continue

            summary = summarize_text(client, text, query, args.model)
            if not summary:
                progress.update(task_id, description=f"[red]Özetlenemedi:[/red] {title[:40]}...", completed=True)
                continue

            progress.update(task_id, description=f"[green]Tamamlandı:[/green] {title[:40]}...", completed=True)
            sources.append({
                "title": title,
                "url": url,
                "summary": summary
            })

    if not sources:
        console.print("[red]Hiçbir kaynaktan geçerli özet üretilemedi. Rapor oluşturulamıyor.[/red]")
        sys.exit(1)

    # 3. Generate Report
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task3 = progress.add_task(description="Kapsamlı Türkçe rapor üretiliyor...", total=None)
        report = generate_report(client, query, sources, args.model)
        progress.update(task3, completed=True)

    if not report:
        console.print("[red]Hata: Rapor üretilemedi.[/red]")
        sys.exit(1)

    # 4. Save Report (Markdown)
    slug = slugify(query)
    output_md = f"{slug}_report.md"
    try:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(report)
        console.print(Panel(f"[bold green]Başarılı![/bold green] Markdown Raporu kaydedildi: [bold cyan]{output_md}[/bold cyan]", border_style="green"))
    except Exception as e:
        console.print(f"[red]Rapor MD dosyaya yazılamadı: {e}[/red]")

    # 5. Save Report (PDF)
    output_pdf = f"{slug}_report.pdf"
    pdf_success = save_pdf(report, output_pdf)
    if pdf_success:
        console.print(Panel(f"[bold green]Başarılı![/bold green] PDF Raporu kaydedildi: [bold cyan]{output_pdf}[/bold cyan]", border_style="green"))
    else:
        console.print("[red]Hata: PDF Raporu oluşturulamadı.[/red]")

    console.print("\n[bold]Rapor Önizleme:[/bold]\n")
    console.print(Markdown(report[:1500] + "\n\n...(devamı raporda)"))

if __name__ == "__main__":
    main()
