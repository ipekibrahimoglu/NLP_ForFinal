import requests
import json
import time
import xml.etree.ElementTree as ET

AIMS_AND_SCOPE = """
Computational Linguistics is the longest-running publication devoted to the
computational and mathematical properties of language and the design and
analysis of natural language processing systems. The journal presents research
on computational linguistics, natural language processing, computational
semantics, machine translation, parsing, generation, discourse, dialogue,
morphology, phonology, grammar induction, language modeling, information
extraction, question answering, and related areas.
"""

START_YEAR = 2015
END_YEAR = 2026
MAX_PER_YEAR = 150
MAX_PAPERS = 1500
OUTPUT_FILE = "data.json"
BASE_URL = "http://export.arxiv.org/api/query"



def fetch_papers():
    papers = []

    for year in range(START_YEAR, END_YEAR, +1):
        start = 0
        batch_size = 100
        year_papers = []

        print("Fetching papers from arXiv (cs.CL)...")

        while len(year_papers) < MAX_PAPERS:
            params = {
                "search_query": f"cat:cs.CL AND submittedDate:[{year}01010000 TO {year}1232359]",
                "start": start,
                "max_results": batch_size,
    
            }

            response = requests.get(BASE_URL, params=params)

            if response.status_code != 200:
                print(f"HTTP Error: {response.status_code}")
                break

            root = ET.fromstring(response.text)
            ns = "{http://www.w3.org/2005/Atom}"
            entries = root.findall(f"{ns}entry")

            if not entries:
                print(f"No more results for {year}.")
                break

            for entry in entries:
                published = entry.find(f"{ns}published")
                if published is None:
                    continue
                year = int(published.text[:4])
                if year < START_YEAR or year > END_YEAR:
                    continue

                summary = entry.find(f"{ns}summary")
                abstract = summary.text.strip() if summary is not None else ""
                if not abstract:
                    continue

                title_el = entry.find(f"{ns}title")
                title = title_el.text.strip() if title_el is not None else ""

                authors = [
                    author.find(f"{ns}name").text
                    for author in entry.findall(f"{ns}author")
                    if author.find(f"{ns}name") is not None
                ]

                id_el = entry.find(f"{ns}id")
                paper_id = id_el.text.strip() if id_el is not None else ""

                year_papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                })

                if len(year_papers) >= MAX_PER_YEAR:
                    break

            print(f"  Collected {len(year_papers)} papers collected for {year}")
            start += batch_size
            time.sleep(3)

        papers.extend(year_papers)

    return papers 

def save(papers):
    output = {
        "journal": "Computational Linguistics (cs.CL on arXiv)",
        "year_range": f"{START_YEAR}-{END_YEAR}",
        "aims_and_scope": AIMS_AND_SCOPE.strip(),
        "total_papers": len(papers),
        "papers": papers,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n Saved {len(papers)} papers to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    papers = fetch_papers()
    save(papers)

    if papers:
        sample = papers[0]
        print(f"\n Sample paper:")
        print(f"  Title: {sample['title']}")
        print(f"  Year:  {sample['year']}")
        print(f"  Abstract: {sample['abstract'][:150]}...")