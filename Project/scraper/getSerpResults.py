import re
from tqdm import tqdm
from fake_useragent import UserAgent
from newspaper import Article, Config
import trafilatura

count = 0

pbar1 = tqdm( total = count, desc = "Progress bar")
pbar2 = tqdm( total = count, desc = "Exception bar")

def scrape_content(url):
    
    global count
    # with lock:
    count += 1
    pbar1.total = count
    pbar1.refresh()
    # try:
    user_agent = UserAgent()
    config = Config()
    config.browser_user_agent = user_agent.random
    config.request_timeout = 10
    article = Article(url, config=config)
    try:
        article.download()  # Set the headers
        article.parse()
        article.nlp()
    except:
        pass
    
    title, text, summary, keywords = "Not Found", "Not Found", "Not Found", "Not Found"
    
    if article:

        get_title = article.title
        get_summary = article.summary
        get_keywords = article.keywords
        if get_title != '':
            title = get_title
        if get_summary != '':
            summary = get_summary
        if get_keywords != '':
            keywords = get_keywords

        # Fetch a content only if above are found
        if title != "Not Found" and summary != "Not Found"and keywords !="Not Found" :
            # code for trafilatura     
            downloaded = trafilatura.fetch_url(url)
            trafilatura_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, output_format= 'txt')
            if trafilatura_text is not None:
                get_text = re.sub(r"\n-", "\n•", trafilatura_text)
                text = get_text
        
        pbar1.update()

    return title, text, summary, keywords