"""
feature_engineering.py — all the editorial-style feature extraction
logic in one place. Each function tags one aspect of the title;
build_features() runs all of them and returns the enriched dataframe.
"""
import re
import pandas as pd

# =====================================================================
# CATEGORY — the desk this belongs to
# =====================================================================
CATEGORY_RULES = {
    "Health":         r"\b(vaccine|measles|flu|disease|hospital|pediatric|health|virus|outbreak|ebola|cholera|hiv)\b",
    "Politics":       r"\b(trump|biden|senate|congress|president|election|gop|republican|democrat|parliament|minister|govt)\b",
    "War & Conflict":  r"\b(war|strike|attack|conflict|kill(?:ed|s)?|bomb|missile|troops|militant|terroris[tm])\b",
    "Business & Finance": r"\b(billion|million|stock|market|bank|economy|gdp|inflation|ipo|trade|tariff)\b",
    "Disaster":       r"\b(earthquake|flood|wildfire|fire|storm|quake|disaster|avalanche)\b",
    "Crime & Legal":  r"\b(court|lawsuit|sues?|charged|arrest|convicted|trial|verdict)\b",
    "Sports":         r"\b(cricket|football|hockey|match|tournament|championship|world cup|fifa|olympic)\b",
    "Technology & AI": r"\b(ai\b|artificial intelligence|openai|chatgpt|robot|tech\b|software|algorithm)\b",
    "Immigration":    r"\b(immigra(?:tion|nt)|migrant|deport|asylum|refugee)\b",
    "Climate & Environment": r"\b(climate|emission|carbon|environment(?:al)?)\b",
    "Entertainment":  r"\b(movie|film|actor|actress|singer|album|celebrity|hollywood)\b",
}

def assign_category(title: str) -> str:
    t = title.lower()
    for category, pattern in CATEGORY_RULES.items():
        if re.search(pattern, t):
            return category
    return "General News"

def assign_subcategory(title: str) -> str:
    t = title.lower()
    primary = assign_category(title)
    for category, pattern in CATEGORY_RULES.items():
        if category != primary and re.search(pattern, t):
            return category
    return "None"


# =====================================================================
# TRENDING TOPIC — computed from the data itself, not a guessed list.
# A "significant word" (proper nouns / capitalized words, 4+ letters,
# not a common English word) that appears in MANY titles across the
# whole dataset is what makes something "trending" in a newsroom sense.
# =====================================================================
COMMON_WORDS_TO_IGNORE = {
    "the", "and", "for", "with", "from", "this", "that", "after", "over",
    "into", "amid", "says", "say", "new", "more", "than", "what", "how",
    "why", "who", "when", "will", "have", "has", "been", "its", "his",
    "her", "their", "are", "was", "were", "not", "but", "out", "off",
}

def _extract_significant_words(title: str) -> set:
    """Words that look like proper nouns/topics: capitalized, 4+ letters."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", title)
    return {
        w.lower() for w in words
        if w[0].isupper() and w.lower() not in COMMON_WORDS_TO_IGNORE
    }

def compute_trending_topics(df: pd.DataFrame, min_occurrences: int = None) -> pd.DataFrame:
    """
    Scans every title's significant words, counts how often each word
    appears across the WHOLE dataset, and flags any title containing a
    word that shows up at least `min_occurrences` times as trending —
    labeling it with that word.

    If min_occurrences isn't given, it scales with dataset size
    (roughly 0.05% of total rows, minimum 3) so this works reasonably
    whether you're running it on 12 rows or 16,000.
    """
    if min_occurrences is None:
        min_occurrences = max(3, int(len(df) * 0.0005))

    word_sets = df["title"].apply(_extract_significant_words)

    from collections import Counter
    counts = Counter()
    for words in word_sets:
        counts.update(words)

    frequent_words = {w for w, c in counts.items() if c >= min_occurrences}

    def label_row(words: set) -> str:
        matches = words & frequent_words
        if not matches:
            return ""
        # pick the most frequent matching word as the label
        best = max(matches, key=lambda w: counts[w])
        return best.title()

    df["trending_topic"] = word_sets.apply(label_row)
    df["is_trending"] = df["trending_topic"].apply(lambda x: "Yes" if x else "No")
    return df


# =====================================================================
# TONE & SEVERITY
# =====================================================================
NEGATIVE_WORDS = {"crisis", "death", "dead", "war", "attack", "warns", "kill", "killed", "fear",
                   "threatens", "collapse", "disaster", "danger", "outbreak", "violence"}
POSITIVE_WORDS = {"wins", "win", "celebrate", "support", "hope", "improve", "breakthrough",
                   "success", "record", "award", "boost"}
SEVERE_WORDS = {"dead", "death", "deaths", "killed", "kills", "die", "dies", "crisis",
                 "disaster", "collapse", "outbreak", "emergency"}
MODERATE_WORDS = {"warns", "threatens", "risk", "concerns", "surge", "rise", "rises"}

def classify_tone(title: str) -> str:
    words = set(re.findall(r"\w+", title.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")

def classify_severity(title: str) -> str:
    words = set(re.findall(r"\w+", title.lower()))
    if words & SEVERE_WORDS:
        return "High"
    if words & MODERATE_WORDS:
        return "Moderate"
    return "Low"


# =====================================================================
# HEADLINE STYLE, URGENCY, SCOPE, STORY TYPE
# =====================================================================
URGENT_WORDS = {"breaking", "urgent", "alert", "warning", "emergency", "immediate", "critical"}
INTERNATIONAL_WORDS = {"un", "global", "world", "international", "nato", "eu"}

def classify_headline_style(title: str) -> str:
    t = title.strip().lower()
    if title.strip().endswith("?"):
        return "Question"
    if any(w in t for w in ["breaking", "urgent", "just in"]):
        return "Breaking News"
    if any(w in t for w in ["explainer", "what we know", "what to know", "why is", "how is"]):
        return "Explainer"
    if any(w in t for w in ["opinion", "analysis"]):
        return "Analysis/Opinion"
    if t.startswith(("in memoriam", "fellows in the news")):
        return "Recurring Feature"
    return "Straight News"

def classify_urgency(title: str) -> str:
    words = set(re.findall(r"\w+", title.lower()))
    if words & URGENT_WORDS:
        return "Urgent"
    if re.search(r"\b(rise|rises|surge|climb|jump|top|reaches?|hits)\b", title.lower()):
        return "Developing"
    return "Routine"

def classify_scope(title: str) -> str:
    t = title.lower()
    words = set(re.findall(r"\w+", t))
    if words & INTERNATIONAL_WORDS or re.search(r"\b(world|global)\b", t):
        return "International"
    if re.search(r"\b(national|country|nationwide|federal)\b", t):
        return "National"
    return "Local/Regional"

def classify_story_type(title: str) -> str:
    t = title.strip().lower()
    if t.startswith("in memoriam") or "dies at" in t or "dies aged" in t:
        return "Obituary"
    if t.startswith("fellows in the news") or t.startswith("health alerts") or t.startswith("fyi"):
        return "Recurring Column"
    if "live:" in t or t.startswith("security council live"):
        return "Live Coverage"
    if t.endswith("?"):
        return "Q&A/Explainer"
    if any(w in t for w in ["recap", "timeline", "what we know", "explained"]):
        return "Explainer"
    return "News Report"


# =====================================================================
# REGION & ENTITIES
# =====================================================================
REGION_KEYWORDS = {
    "US & North America": r"\b(us|u\.s\.|america|washington|congress|white house|canada)\b",
    "Middle East":       r"\b(iran|iraq|israel|gaza|saudi|syria|lebanon|yemen|hormuz)\b",
    "South Asia":        r"\b(pakistan|india|kashmir|afghanistan|bangladesh)\b",
    "Europe":            r"\b(russia|ukraine|france|spain|germany|uk|britain|eu\b)\b",
    "East & Southeast Asia": r"\b(china|japan|korea|taiwan|myanmar)\b",
    "Africa":            r"\b(africa|kenya|nigeria|sudan|uganda|libya)\b",
    "Latin America":     r"\b(mexico|brazil|peru|argentina|venezuela|honduras)\b",
}

KNOWN_ENTITIES = [
    "Trump", "Biden", "Netanyahu", "Putin", "Zelensky", "Iran", "Israel", "Gaza",
    "Ukraine", "Russia", "China", "Pakistan", "India", "AAP", "CDC", "FDA", "FIFA",
    "OpenAI", "Anthropic", "UN", "WHO", "NATO", "EU",
]

def assign_region(title: str) -> str:
    t = title.lower()
    for region, pattern in REGION_KEYWORDS.items():
        if re.search(pattern, t):
            return region
    return "Global/Unspecified"

def find_entities(title: str) -> str:
    found = [e for e in KNOWN_ENTITIES if re.search(rf"\b{re.escape(e)}\b", title, re.IGNORECASE)]
    return ", ".join(found)


# =====================================================================
# AUDIENCE
# =====================================================================
CHILD_WORDS = {"child", "children", "kids", "infant", "infants", "pediatric",
               "pediatrics", "baby", "babies", "toddler", "teen", "teens", "youth", "youths"}

def involves_children(title: str) -> str:
    words = set(re.findall(r"\w+", title.lower()))
    return "Yes" if words & CHILD_WORDS else "No"

def classify_audience(title: str) -> str:
    t = title.lower()
    if re.search(r"\b(parents|caregivers|families|pediatric|children)\b", t):
        return "Parents & Families"
    if re.search(r"\b(policy|congress|senate|regulation|govt|government)\b", t):
        return "Policy Makers"
    if re.search(r"\b(investors|market|stock|shares|ipo)\b", t):
        return "Investors"
    if re.search(r"\b(physicians|clinicians|pediatricians|doctors)\b", t):
        return "Medical Professionals"
    return "General Public"


# =====================================================================
# ORCHESTRATOR — runs every tagging function and adds all columns
# =====================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["word_count"] = df["title"].apply(lambda t: len(t.split()))

    df["category"] = df["title"].apply(assign_category)
    df["subcategory"] = df["title"].apply(assign_subcategory)

    df = compute_trending_topics(df)

    df["tone"] = df["title"].apply(classify_tone)
    df["severity"] = df["title"].apply(classify_severity)

    df["headline_style"] = df["title"].apply(classify_headline_style)
    df["urgency"] = df["title"].apply(classify_urgency)
    df["story_scope"] = df["title"].apply(classify_scope)
    df["story_type"] = df["title"].apply(classify_story_type)
    df["contains_quote"] = df["title"].apply(
        lambda t: "Yes" if re.search(r"[\"\u2018\u2019]", t) else "No"
    )
    df["contains_statistic"] = df["title"].apply(
        lambda t: "Yes" if re.search(r"\d+%|\$\d|\b\d{2,}\b", t) else "No"
    )

    df["region"] = df["title"].apply(assign_region)
    df["mentioned_entities"] = df["title"].apply(find_entities)

    df["involves_children"] = df["title"].apply(involves_children)
    df["audience"] = df["title"].apply(classify_audience)

    return df