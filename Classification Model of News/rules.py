from utils import path_segments, domain, contains, normalize

# ============================================================
# URL TAXONOMY
# Only for label correction.
# URL is NEVER used as model input.
# ============================================================

def url_taxonomy(url):
    seg = path_segments(url)
    dom = domain(url)

    ordered = [
        ("Health", {
            "health-news",
            "health",
            "medicine",
            "healthcare",
            "pharmaceuticals",
            "biotechnology",
        }),

        ("Technology", {
            "technology",
            "info-tech",
            "big-tech",
        }),

        ("Politics", {
            "politics",
            "government",
            "elections",
            "policy",
            "international-relations",
            "geopolitics",
        }),

        ("Markets", {
            "markets",
            "market-data",
            "equities",
            "stock-markets",
            "forex",
            "ipos",
            "rates",
            "mortgages",
            "capital-markets-currencies",
            "technical-analysis",
        }),

        ("Energy", {
            "energy",
            "energy-and-climate",
            "renewables",
        }),

        ("Business", {
            "companies",
            "business-deals",
        }),
    ]

    for label, sections in ordered:
        if any(s in sections for s in seg[:5]):
            return label

    if dom == "money.usnews.com":
        if "investing" in seg or "loans" in seg:
            return "Markets"

    if dom == "fool.com":
        if "investing" in seg:
            return "Markets"

    if dom == "barrons.com":
        if seg and seg[0] == "advisor":
            return "Business"

    if dom == "thecanadianpressnews.ca":
        if seg and seg[0] in {"world", "national", "politics"}:
            return "Politics"
        if seg and seg[0] == "business":
            return "Business"

    return None


# ============================================================
# TRUSTED TITLE RULES
# Based on patterns actually present in this dataset.
# ============================================================

def trusted_title_label(title):
    t = " " + normalize(title) + " "

    # --------------------------------------------------------
    # MARKETS
    # Explicit securities / price / investing language.
    # --------------------------------------------------------
    if contains(t, [
        r"\bstocks?\b",
        r"\bshares?\b",
        r"\bwall street\b",
        r"\bnasdaq\b",
        r"\bdow jones\b",
        r"\bs&p\s*500\b",
        r"\betfs?\b",
        r"\bbitcoin\b",
        r"\bethereum\b",
        r"\bcrypto\w*\b",
        r"\bforex\b",
        r"\bbonds?\b",
        r"\byields?\b",
        r"\bdividends?\b",
        r"\bprice target\b",
        r"\bupgrade(?:d|s)?\b",
        r"\bdowngrade(?:d|s)?\b",
        r"\bfutures\b",
    ]):
        return "Markets"

    if contains(t, [
        r"\boil prices?\b",
        r"\bcrude prices?\b",
        r"\bgold prices?\b",
        r"\bsilver prices?\b",
        r"\bmortgage rates?\b",
        r"\brate cuts?\b",
        r"\brate hikes?\b",
        r"\binterest rates?\b",
    ]):
        return "Markets"

    # --------------------------------------------------------
    # HEALTH - expanded from actual dataset
    # --------------------------------------------------------
    if contains(t, [
        r"\bcancer\b",
        r"\bclinical trials?\b",
        r"\bphase\s*(?:1|2|3)(?:a|b)?\b",
        r"\bpatients?\b",
        r"\bvaccines?\b",
        r"\btherapy\b",
        r"\btreatments?\b",
        r"\bdiseases?\b",
        r"\bhospitals?\b",
        r"\bfda\b",
        r"\balzheimer",
        r"\bobesity\b",
        r"\bmedical\b",
        r"\bmedicine\b",
        r"\bpharma(?:ceutical|ther)?\w*\b",
        r"\bbiopharma\w*\b",
        r"\bdiagnostic\w*\b",
        r"\bdiabetes\b",
        r"\bmedicare\b",
        r"\bprostate\b",
        r"\bkidney\b",
        r"\brheumatoid\b",
        r"\bbiomarker\w*\b",
        r"\bopioid\b",
        r"\btherapeutic designation\b",
    ]):
        return "Health"

    # --------------------------------------------------------
    # POLITICS
    # --------------------------------------------------------
    if contains(t, [
        r"\bgovernment\b",
        r"\bpresident\b",
        r"\bprime minister\b",
        r"\belections?\b",
        r"\bparliament\b",
        r"\bsenate\b",
        r"\bcongress\b",
        r"\bsupreme court\b",
        r"\bsanctions?\b",
        r"\bcease[- ]?fire\b",
        r"\bmilitary\b",
        r"\bminister\b",
        r"\blegislation\b",
        r"\btariffs?\b",
        r"\bwhite house\b",
        r"\bshutdown\b",
    ]):
        return "Politics"

    # --------------------------------------------------------
    # BUSINESS
    # Corporate operations / deals / executives.
    # --------------------------------------------------------
    if contains(t, [
        r"\bacquir\w*\b",
        r"\bmergers?\b",
        r"\btakeovers?\b",
        r"\bceo\b",
        r"\bcfo\b",
        r"\bchief executive\b",
        r"\bappoint\w*\b",
        r"\bpartnerships?\b",
        r"\bcontracts?\b",
        r"\bdivest\w*\b",
        r"\bspin[- ]?off\b",
        r"\brestructur\w*\b",
        r"\blayoffs?\b",
    ]):
        return "Business"

    # --------------------------------------------------------
    # ENERGY - expanded from actual mining/resource headlines
    # --------------------------------------------------------
    if contains(t, [
        r"\boil production\b",
        r"\bgas production\b",
        r"\boil fields?\b",
        r"\blng\b",
        r"\bnuclear reactors?\b",
        r"\bnuclear power\b",
        r"\bsolar\b",
        r"\bwind power\b",
        r"\brenewables?\b",
        r"\buranium\b",
        r"\bpipelines?\b",
        r"\bdrilling\b",
        r"\bpower grid\b",

        # Actual mining/resource patterns in dataset
        r"\bdrill (?:hole|program|rig)\b",
        r"\bdrills?\s+\d",
        r"\bmineralization\b",
        r"\bmine life\b",
        r"\bwellfield\b",
        r"\bore\b",
        r"\bgraphite\b",
        r"\blithium\b",
        r"\brare earths?\b",
        r"\bcopper (?:project|mine|deposit|skarn)\b",
        r"\bgold (?:project|mine|deposit)\b",
    ]):
        return "Energy"

    # --------------------------------------------------------
    # TECHNOLOGY - expanded from actual dataset
    # --------------------------------------------------------
    if contains(t, [
        r"\bartificial intelligence\b",
        r"\bgenerative ai\b",
        r"\bai\b",
        r"\bchatgpt\b",
        r"\bopenai\b",
        r"\bsemiconductors?\b",
        r"\bchips?\b",
        r"\bgpus?\b",
        r"\bsoftware\b",
        r"\bcyber\w*\b",
        r"\bcloud\b",
        r"\brobot\w*\b",
        r"\bquantum\b",
        r"\bdata cent(?:er|re)s?\b",

        # Actual missing technology patterns
        r"\bdatabase\b",
        r"\bazure\b",
        r"\bdata leak\b",
        r"\bdata breach\b",
        r"\bwi-?fi\b",
        r"\bdriverless\b",
        r"\bautonomous (?:car|vehicle|driving|taxi)\b",
        r"\b5g\b",
        r"\bbroadband\b",
        r"\binternet services\b",
    ]):
        return "Technology"

    return None
