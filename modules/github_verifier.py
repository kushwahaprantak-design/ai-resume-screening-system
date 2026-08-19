import re
import urllib.request
import json

def extract_github_username(text_or_url: str) -> str:
    """
    Extract GitHub username from text or URL using RegEx.
    """
    if not text_or_url:
        return None
        
    pattern = r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, text_or_url, re.IGNORECASE)
    if match:
        username = match.group(1).lower()
        if username not in ["orgs", "topics", "features", "explore", "trending"]:
            return username
    return None


def verify_github_profile(github_url_or_text: str) -> dict:
    """
    Call GitHub REST API with graceful timeout & rate limit handling to calculate Credibility Index.
    """
    username = extract_github_username(github_url_or_text)
    if not username:
        return {
            "username": None,
            "credibility_index": "Unverified",
            "badge_color": "#94A3B8",
            "public_repos": 0,
            "followers": 0,
            "languages": [],
            "status_text": "No GitHub link found in resume"
        }

    try:
        url = f"https://api.github.com/users/{username}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AI-Resume-Screening-System/1.0"}
        )
        
        with urllib.request.urlopen(req, timeout=2.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                public_repos = data.get("public_repos", 0)
                followers = data.get("followers", 0)

                if public_repos >= 5 or followers >= 10:
                    credibility_index = "High Credibility"
                    badge_color = "#10B981" # Green
                elif public_repos >= 1:
                    credibility_index = "Moderate Credibility"
                    badge_color = "#F59E0B" # Amber
                else:
                    credibility_index = "Low Activity"
                    badge_color = "#64748B"

                return {
                    "username": username,
                    "credibility_index": credibility_index,
                    "badge_color": badge_color,
                    "public_repos": public_repos,
                    "followers": followers,
                    "profile_url": f"https://github.com/{username}",
                    "status_text": f"Verified profile with {public_repos} repos"
                }

    except Exception as e:
        print(f"GitHub API check for '{username}' skipped/failed: {e}")

    # Fallback if API timeout or offline
    return {
        "username": username,
        "credibility_index": "Unverified (API Timeout)",
        "badge_color": "#94A3B8",
        "public_repos": 0,
        "followers": 0,
        "profile_url": f"https://github.com/{username}",
        "status_text": f"Profile link found: github.com/{username}"
    }
