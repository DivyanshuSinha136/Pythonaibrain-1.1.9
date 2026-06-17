import requests
import wikipedia
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string

stop_words = set(stopwords.words('english'))

def search_duckduckgo(query: str) -> list:
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    real_links = []
    for a in soup.find_all('a', class_='result__a', limit=3):
        href = a.get('href', '')
        if href.startswith('//'):
            href = 'https:' + href

        parsed = urlparse(href)
        query_params = parse_qs(parsed.query)
        if 'uddg' in query_params:
            real_url = query_params['uddg'][0]
            real_url = unquote(real_url)
            real_links.append(real_url)
        else:
            real_links.append(href)  # fallback

    return real_links

def get_page_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        return text
    except requests.exceptions.RequestException:
        return ""

def summarize_text(text: str, num_sentences: int = 5) -> list:
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    freq = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for w in words:
            if w in stop_words or w in string.punctuation:
                continue
            freq[w] = freq.get(w, 0) + 1

    sentence_scores = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score = sum(freq.get(w, 0) for w in words)
        sentence_scores.append((score, sentence))

    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in sentence_scores[:num_sentences]]
    return top_sentences

class Search:
    def __init__(self, query: str = 'pythonaibrain'):
        self.query: str = query
        self.lst_links: List[str] = []
        self.responses: List[str] = []
        self.lst_preview: List[List[str]] = []

    def run(self) -> None:
        self.lst_links = search_duckduckgo(self.query)
        self.responses = [get_page_text(link) for link in self.lst_links]
        self.lst_preview = [summarize_text(resp) for resp in self.responses]

    def display_results(self) -> None:
        links = []
        summarys = []
        if not self.lst_links:
            #print("No results found.")
            return "No results found.", None

        for i, (link, summary) in enumerate(zip(self.lst_links, self.lst_preview), 1):
            #print(f"\n{i}. {link}")
            #print("Summary:")
            links.append(link)
            if summary:
                #print("\n".join(summary))
                summarys.append(summary)
            else:
                try:
                    summaries = self.get_wikipedia_info(self.query)
                    if summaries == "Multiple pages found. Please provide more context.":
                        return None

                    elif summaries == "Page not found.":
                        return "No readable content or summary available."

                    else:
                        return summaries

                except:
                    print("No readable content or summary available.")

        return link, summarys

    def get_results_str(self) -> str:
        if not self.lst_links:
            return "No results found."

        results = []
        for i, (link, summary) in enumerate(zip(self.lst_links, self.lst_preview), 1):
            result_str = f"{i}. {link}\nSummary:\n"
            if summary:
                result_str += "\n".join(summary)
            else:
                result_str += "No readable content or summary available."
            results.append(result_str)

        return "\n\n".join(results)
    def get_wikipedia_info(self, topic):
        try:
            summary = wikipedia.summary(topic, sentences=5)
            return summary
        except wikipedia.exceptions.DisambiguationError:
            return "Multiple pages found. Please provide more context."
        except wikipedia.exceptions.PageError:
            return "Page not found."
