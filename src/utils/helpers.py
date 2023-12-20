import re

def filter_citations_and_links(text):
    # Remove citations like [1], [2], ...
    text_no_citations = re.sub(r'\[\d+\]', '', text)

    # Remove links
    text_no_links = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           '', text_no_citations)

    # Remove www links
    text_no_links = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           '', text_no_links)

    return text_no_links