import re
from urllib.parse import urlparse


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())


def contains(text, patterns):
    return any(re.search(p, text, flags=re.I) for p in patterns)


def path_segments(url):
    try:
        return [
            s for s in urlparse(str(url)).path.lower().split("/")
            if s
        ]
    except Exception:
        return []


def domain(url):
    try:
        return (
            urlparse(str(url))
            .netloc.lower()
            .replace("www.", "")
        )
    except Exception:
        return ""
