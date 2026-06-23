import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from google.colab import userdata
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN_READ')

response = requests.get('https://australianpremiumsolar.co.in/blog/')
context = BeautifulSoup(response.text, 'html.parser')

for script in context(["script", "style", "svg", "noscript"]):
    script.extract()

context = context.prettify()


def pre_process_and_split_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    
    # Remove heavy noise that carries zero schema value
    for script_or_style in soup(["script", "style", "noscript", "svg"]):
        script_or_style.decompose()

    # ----------------------------------------------------
    # SEGMENT 1: METADATA CHUNK
    # ----------------------------------------------------
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
        
    # Find JSON-LD schemas
    json_ld = soup.find_all('script', type='application/ld+json')
    for i, jld in enumerate(json_ld):
        metadata_lines.append(f"JSON-LD Schema [{i}]: {jld.string.strip() if jld.string else 'Present but empty'}")

    metadata_chunk = "\n".join(metadata_lines)

    # ----------------------------------------------------
    # SEGMENT 2: NAVIGATION & STRUCTURE CHUNK
    # ----------------------------------------------------
    nav_lines = []
    # Targeted extraction of nav containers
    nav_elements = soup.find_all(['nav', 'header', 'footer', 'aside'])
    
    # Also grab elements with common structural class/id names
    structural_selectors = soup.find_all(class_=re.compile(r'nav|menu|sidebar|footer|header|breadcrumb|pagination', re.I))
    
    combined_nav_nodes = set(nav_elements + structural_selectors)
    
    for node in combined_nav_nodes:
        # Extract links inside these structural nodes
        links = node.find_all('a')
        for link in links:
            text = link.get_text(strip=True)
            href = link.get('href')
            parent_type = node.name
            parent_class = node.get('class', '')
            nav_lines.append(f"[{parent_type} ({parent_class})] Text: {text} | URL: {href}")
            
    # Deduplicate lines while preserving order
    nav_chunk = "\n".join(list(dict.fromkeys(nav_lines)))

    # ----------------------------------------------------
    # SEGMENT 3: INFORMATION & CONTENT CHUNK
    # ----------------------------------------------------
    content_lines = []
    
    # Strip the nav/footer blocks out so they don't pollute the content chunk
    for block in combined_nav_nodes:
        try:
            block.decompose()
        except ValueError:
            pass # Already removed
            
    # Extract structural headings, body text, tables, CTAs, and media
    content_tags = soup.find_all(['h1', 'h2', 'h3', 'p', 'table', 'ul', 'ol', 'img', 'button', 'a'])
    
    for tag in content_tags:
        if tag.name in ['h1', 'h2', 'h3']:
            content_lines.append(f"\n{tag.name.upper()}: {tag.get_text(strip=True)}")
        elif tag.name == 'p':
            txt = tag.get_text(strip=True)
            if txt: content_lines.append(f"Paragraph: {txt}")
        elif tag.name in ['ul', 'ol']:
            items = [li.get_text(strip=True) for li in tag.find_all('li') if li.get_text(strip=True)]
            if items: content_lines.append(f"List Items: { ' | '.join(items) }")
        elif tag.name == 'table':
            content_lines.append(f"Table Data: {tag.get_text(space=' ', strip=True)}")
        elif tag.name == 'img':
            content_lines.append(f"Image Source: {tag.get('src')} | Alt: {tag.get('alt', 'None')}")
        elif tag.name in ['button'] or (tag.name == 'a' and ('btn' in "".join(tag.get('class', '')) or 'button' in "".join(tag.get('class', '')))):
            content_lines.append(f"CTA Button Text: {tag.get_text(strip=True)} | Link/Action: {tag.get('href', 'Form Action')}")

    content_chunk = "\n".join(content_lines)

    return {
        "metadata": metadata_chunk,
        "navigation": nav_chunk,
        "content": content_chunk
    }

content = pre_process_and_split_html(context)
print(len(content))

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
Strict Rules for Execution:
1. Do not include raw HTML tags in your final text document unless specifically requested as an example.
2. If a section or element is missing from the provided code (e.g., no sidebar or no JSON-LD), explicitly state "None found" under that specific subheading.
3. Maintain the original semantic meaning of the content without adding external assumptions.

Here is the scraped webpage code:
{context}""")
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    temperature = 0.1,
    max_new_tokens = 3000 # Restrict this so (input + output) <= 8192
)

bot = ChatHuggingFace(llm=llm)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300,
    length_function=len
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
chromadb = Chroma()


# 2. Update your processing loop
for segment_name, segment_text in content.items():
    print(f"\n==================================================")
    print(f"--- PROCESSING SEGMENT: {segment_name.upper()} ---")
    print(f"==================================================")
    
    if not segment_text.strip():
        print("None found\n")
        continue

    # Split this specific segment's text into small chunks
    sub_chunks = text_splitter.split_text(segment_text)
    print(f"Segment split into {len(sub_chunks)} sub-chunks due to length constraints.\n")

    chain = prompt | bot

    # Process each sub-chunk through the LLM
    for idx, chunk in enumerate(sub_chunks):
        if len(sub_chunks) > 1:
            print(f"--- Processing Sub-chunk {idx + 1}/{len(sub_chunks)} ---")
        
        try:          
            para = chain.invoke({'context': chunk})

            output_text = para.content if hasattr(para, 'content') else para
            chromadb.add_texts(texts=[output_text], metadatas=[{'segment': segment_name}])
            print(output_text)
            print("-" * 30)
            
        except Exception as e:
            print(f"Error processing sub-chunk {idx + 1}: {e}")

