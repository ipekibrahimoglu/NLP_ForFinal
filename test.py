import requests

r = requests.get("http://export.arxiv.org/api/query?search_query=cat:cs.CL&max_results=3")
print(r.status_code)
print(r.text[:500])