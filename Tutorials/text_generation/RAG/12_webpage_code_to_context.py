import os
import re
import hashlib
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.colab import userdata

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
os.environ['HF_TOKEN'] = hf_token


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SITEMAP FETCHER
# Returns a list of URLs from an XML sitemap (handles sitemap indexes too).
# ══════════════════════════════════════════════════════════════════════════════
def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Parse a sitemap XML and return all <loc> URLs, including nested sitemaps."""
    resp = requests.get(sitemap_url, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    # Sitemap index — recurse into each child sitemap
    if root.tag.endswith('sitemapindex'):
        urls = []
        for loc in root.findall('sm:sitemap/sm:loc', ns):
            urls.extend(fetch_sitemap_urls(loc.text.strip()))
        return urls

    # Regular sitemap
    return [loc.text.strip() for loc in root.findall('sm:url/sm:loc', ns)]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — HTML SCRAPER
# ══════════════════════════════════════════════════════════════════════════════
def fetch_and_clean_html(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(["script", "style", "svg", "noscript"]):
        tag.extract()
    return soup.prettify()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SEGMENTED HTML PARSER  (unchanged from original)
# Returns dict with keys: "metadata", "navigation", "content"
# ══════════════════════════════════════════════════════════════════════════════
def pre_process_and_split_html(raw_html: str) -> dict[str, str]:
    soup = BeautifulSoup(raw_html, 'html.parser')
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # ── SEGMENT 1: METADATA ──────────────────────────────────────────────────
    metadata_lines = []
    if soup.title:
        metadata_lines.append(f"Title: {soup.title.string}")
    for meta in soup.find_all('meta'):
        attrs = meta.attrs
        if 'name' in attrs or 'property' in attrs:
            key = attrs.get('name') or attrs.get('property')
            val = attrs.get('content')
            metadata_lines.append(f"Meta {key}: {val}")
    for link in soup.find_all('link', rel=re.compile(r'canonical|alternate', re.I)):
        metadata_lines.append(f"Link {link.get('rel')}: {link.get('href')}")
    for i, jld in enumerate(soup.find_all('script', type='application/ld+json')):
        metadata_lines.append(
            f"JSON-LD Schema [{i}]: {jld.string.strip() if jld.string else 'Present but empty'}"
        )
    metadata_chunk = "\n".join(metadata_lines)

    # ── SEGMENT 2: NAVIGATION ────────────────────────────────────────────────
    nav_lines = []
    nav_elements = soup.find_all(['nav', 'header', 'footer', 'aside'])
    structural_selectors = soup.find_all(
        class_=re.compile(r'nav|menu|sidebar|footer|header|breadcrumb|pagination', re.I)
    )
    combined_nav_nodes = set(nav_elements + structural_selectors)
    for node in combined_nav_nodes:
        for link in node.find_all('a'):
            text = link.get_text(strip=True)
            href = link.get('href')
            nav_lines.append(
                f"[{node.name} ({node.get('class', '')})] Text: {text} | URL: {href}"
            )
    nav_chunk = "\n".join(list(dict.fromkeys(nav_lines)))

    # ── SEGMENT 3: CONTENT ───────────────────────────────────────────────────
    content_lines = []
    for block in combined_nav_nodes:
        try:
            block.decompose()
        except ValueError:
            pass
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'table', 'ul', 'ol', 'img', 'button', 'a']):
        if tag.name in ['h1', 'h2', 'h3']:
            content_lines.append(f"\n{tag.name.upper()}: {tag.get_text(strip=True)}")
        elif tag.name == 'p':
            txt = tag.get_text(strip=True)
            if txt:
                content_lines.append(f"Paragraph: {txt}")
        elif tag.name in ['ul', 'ol']:
            items = [li.get_text(strip=True) for li in tag.find_all('li') if li.get_text(strip=True)]
            if items:
                content_lines.append(f"List Items: {' | '.join(items)}")
        elif tag.name == 'table':
            content_lines.append(f"Table Data: {tag.get_text(separator=' ', strip=True)}")
        elif tag.name == 'img':
            content_lines.append(f"Image Source: {tag.get('src')} | Alt: {tag.get('alt', 'None')}")
        elif tag.name in ['button'] or (
            tag.name == 'a' and ('btn' in "".join(tag.get('class', '')) or
                                  'button' in "".join(tag.get('class', '')))
        ):
            content_lines.append(
                f"CTA Button Text: {tag.get_text(strip=True)} | Link/Action: {tag.get('href', 'Form Action')}"
            )
    content_chunk = "\n".join(content_lines)

    return {"metadata": metadata_chunk, "navigation": nav_chunk, "content": content_chunk}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DEDUPLICATION ENGINE  ◄── THE NEW CORE LOGIC
#
# How it works:
#   • Each segment text is split into "atomic blocks" — individual lines or
#     small coherent units (e.g. one nav link, one meta tag, one paragraph).
#   • Every block gets an MD5 fingerprint.
#   • A global seen_hashes set persists across ALL pages in the crawl.
#   • Before feeding any block to the LLM, we check its hash:
#       - SEEN   → skip (already processed from a prior page)
#       - UNSEEN → pass through + add hash to seen set
#   • The filtered text is what gets sent to the LLM.
#
# Why line-level instead of whole-segment hashing?
#   Hashing the entire nav chunk would miss partial overlaps.  A page might
#   share 90 % of its nav but have one unique breadcrumb — we want to keep
#   just that breadcrumb, not re-send the other 90 %.
# ══════════════════════════════════════════════════════════════════════════════

class DeduplicationEngine:
    def __init__(self):
        # Persists for the entire crawl session
        self.seen_hashes: set[str] = set()
        # Tracks how many blocks were saved across the crawl
        self.stats = {"total_blocks": 0, "skipped_blocks": 0, "sent_blocks": 0}

    def _hash(self, text: str) -> str:
        """Stable MD5 fingerprint of a normalised text block."""
        normalised = re.sub(r'\s+', ' ', text).strip().lower()
        return hashlib.md5(normalised.encode('utf-8')).hexdigest()

    def _split_into_blocks(self, segment_text: str) -> list[str]:
        """
        Split a segment into atomic blocks for fine-grained dedup.
        Each non-empty line is one block (nav links, meta tags, paragraphs).
        Adjacent heading + paragraph pairs are kept together so context
        is not lost when a heading is shared but the paragraph beneath is new.
        """
        blocks, buffer = [], []
        for line in segment_text.splitlines():
            stripped = line.strip()
            if not stripped:
                if buffer:
                    blocks.append("\n".join(buffer))
                    buffer = []
                continue
            # Group headings with the next line so we never orphan a heading
            if stripped.startswith(("H1:", "H2:", "H3:")):
                if buffer:
                    blocks.append("\n".join(buffer))
                    buffer = []
                buffer.append(stripped)
            else:
                buffer.append(stripped)
                # Flush after each non-heading line to keep blocks small
                blocks.append("\n".join(buffer))
                buffer = []
        if buffer:
            blocks.append("\n".join(buffer))
        return [b for b in blocks if b.strip()]

    def filter_segment(self, segment_text: str, segment_name: str, page_url: str) -> str:
        """
        Returns only the NOVEL (previously unseen) portion of a segment.
        Mutates self.seen_hashes in place.
        """
        blocks = self._split_into_blocks(segment_text)
        novel_blocks = []
        skipped, kept = 0, 0

        for block in blocks:
            self.stats["total_blocks"] += 1
            h = self._hash(block)
            if h in self.seen_hashes:
                skipped += 1
                self.stats["skipped_blocks"] += 1
            else:
                self.seen_hashes.add(h)
                novel_blocks.append(block)
                kept += 1
                self.stats["sent_blocks"] += 1

        print(
            f"  [{segment_name}] {kept} novel blocks kept, "
            f"{skipped} duplicate blocks skipped."
        )
        return "\n".join(novel_blocks)

    def print_summary(self):
        s = self.stats
        saved_pct = (s['skipped_blocks'] / s['total_blocks'] * 100) if s['total_blocks'] else 0
        print("\n" + "═" * 60)
        print("DEDUPLICATION SUMMARY")
        print("═" * 60)
        print(f"  Total blocks seen  : {s['total_blocks']}")
        print(f"  Novel blocks sent  : {s['sent_blocks']}")
        print(f"  Duplicate skipped  : {s['skipped_blocks']} ({saved_pct:.1f}% token savings)")
        print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — LLM + VECTOR STORE SETUP  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
prompt = PromptTemplate.from_template(
    """You are an expert web scraping analyst and data architect. I am going to provide you with the source code (HTML/text) of a scraped webpage.

Your task is to analyze this code and generate a comprehensive, structured text document that extracts and categorizes everything related to the page's Navigation, Information, and Metadata.

Please organize the output exactly according to the following schema:

### 1. METADATA
Extract all high-level and hidden configuration data. Include:
- Page Title (from <title> tag)
- Meta Description & Keywords
- Open Graph (og:) or Twitter card tags (for social sharing)
- Canonical URL
- Language, charset, and viewport settings
- Any embedded schema.org / JSON-LD data found (summarize the type and key entities)

### 2. NAVIGATION & STRUCTURE
Map out how a user moves through this page and the wider site based on the links and menus present. Include:
- Main Header Navigation (List text and destination URLs for main menu items)
- Sidebar or Contextual Menus
- Footer Navigation Links
- Breadcrumbs (if present)
- Pagination details (e.g., "Next", "Previous", or page numbers)
- Notable internal vs. external links found in these structural areas

### 3. INFORMATION & CONTENT
Extract the core value of the page. Do not just dump text; organize it hierarchically:
- Primary Heading (H1) and Subheadings (H2, H3) to show content flow.
- Main Body Content (Summarize or extract the primary textual paragraphs, articles, or product descriptions).
- Key Data Points / Tables / Lists (Capture structured informational blocks).
- Media Assets (List important images, videos, or audio elements with their alt text and source URLs).
- Call-to-Action (CTA) elements (Buttons, forms, or sign-up links, including their text and purposes).

---
Strict Rules:
1. Do not include raw HTML tags in your final output.
2. If a section is missing, explicitly state "None found" under that subheading.
3. Maintain the original semantic meaning without external assumptions.

Here is the scraped webpage code:
{context}"""
)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation',
    temperature=0.1,
    max_new_tokens=3000,
)
bot = ChatHuggingFace(llm=llm)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
chromadb = Chroma()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MAIN CRAWL LOOP
# ══════════════════════════════════════════════════════════════════════════════
def process_segment_through_llm(segment_text: str, segment_name: str, page_url: str):
    """Chunk → LLM → ChromaDB for a single already-deduplicated segment."""
    sub_chunks = text_splitter.split_text(segment_text)
    print(f"  → Sending {len(sub_chunks)} sub-chunk(s) to LLM.")
    chain = prompt | bot

    for idx, chunk in enumerate(sub_chunks):
        if len(sub_chunks) > 1:
            print(f"     Sub-chunk {idx + 1}/{len(sub_chunks)}")
        try:
            result = chain.invoke({'context': chunk})
            output_text = result.content if hasattr(result, 'content') else result
            chromadb.add_texts(
                texts=[output_text],
                metadatas=[{'segment': segment_name, 'source_url': page_url}],
            )
            print(output_text)
            print("-" * 40)
        except Exception as e:
            print(f"  ⚠ Error on sub-chunk {idx + 1}: {e}")


def crawl_sitemap(sitemap_url: str):
    """
    Main entry point.
    1. Fetch all URLs from the sitemap.
    2. For each URL, scrape → segment → deduplicate → LLM → store.
    """
    dedup = DeduplicationEngine()

    print(f"Fetching sitemap: {sitemap_url}")
    urls = fetch_sitemap_urls(sitemap_url)
    print(f"Found {len(urls)} URLs to process.\n")

    for page_num, url in enumerate(urls, start=1):
        print(f"\n{'═' * 60}")
        print(f"PAGE {page_num}/{len(urls)}: {url}")
        print('═' * 60)

        # ── 1. Fetch & parse ──────────────────────────────────────────────
        try:
            raw_html = fetch_and_clean_html(url)
        except Exception as e:
            print(f"  ⚠ Could not fetch page: {e}")
            continue

        segments = pre_process_and_split_html(raw_html)

        # ── 2. Deduplication (runs BEFORE the LLM) ────────────────────────
        print("\n  [ DEDUPLICATION PASS ]")
        novel_segments = {}
        for seg_name, seg_text in segments.items():
            if not seg_text.strip():
                print(f"  [{seg_name}] Empty — skipping.")
                continue
            novel_text = dedup.filter_segment(seg_text, seg_name, url)
            if novel_text.strip():
                novel_segments[seg_name] = novel_text
            else:
                print(f"  [{seg_name}] 100% duplicate — skipping LLM entirely.")

        # ── 3. LLM + storage (only novel content) ─────────────────────────
        if not novel_segments:
            print("\n  No novel content on this page. Skipping LLM call entirely.")
            continue

        print("\n  [ LLM PROCESSING ]")
        for seg_name, novel_text in novel_segments.items():
            print(f"\n  --- Segment: {seg_name.upper()} ---")
            process_segment_through_llm(novel_text, seg_name, url)

    # ── Final stats ───────────────────────────────────────────────────────
    dedup.print_summary()


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SITEMAP_URL = "https://australianpremiumsolar.co.in/sitemap.xml"
    crawl_sitemap(SITEMAP_URL)
