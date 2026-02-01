import os
import datetime
import glob
import json
import google.generativeai as genai
from slugify import slugify
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
TOPIC_BASE_DIR = "topic/nlp"
CURRICULUM_FILE = "curriculum.json"

def get_existing_topic_slugs():
    """Scans the directory for existing .md files and returns their slugs."""
    slugs = set()
    files = glob.glob(os.path.join(TOPIC_BASE_DIR, "**/*.md"), recursive=True)
    for f in files:
        # Filename without extension is the slug
        slug = os.path.basename(f).replace(".md", "")
        slugs.add(slug)
    return slugs

def get_next_topic():
    """Reads curriculum.json and returns the first topic that hasn't been generated."""
    if not os.path.exists(CURRICULUM_FILE):
        print(f"Error: {CURRICULUM_FILE} not found.")
        return None

    with open(CURRICULUM_FILE, 'r') as f:
        data = json.load(f)
        curriculum_topics = data.get("topics", [])

    existing_slugs = get_existing_topic_slugs()

    for topic in curriculum_topics:
        slug = slugify(topic)
        if slug not in existing_slugs:
            print(f"Next topic selected: {topic}")
            return topic
    
    print("All topics in the curriculum have been covered!")
    return None

def get_lesson_info():
    now = datetime.datetime.now()
    # ISO Week number
    week_num = now.isocalendar()[1]
    # Date string for folder
    date_str = now.strftime("%Y-%m-%d")
    # Lesson number (every 4 hours: 0, 4, 8, 12, 16, 20) -> 1, 2, 3, 4, 5, 6
    lesson_num = (now.hour // 4) + 1
    return week_num, date_str, lesson_num, now

def generate_content():
    topic_name = get_next_topic()
    if not topic_name:
        return None

    prompt = f"""
Teach me about the following topic in Natural Language Processing: "{topic_name}".

Please strictly follow this format:
Topic: {topic_name}

1- Provide formal definition, what is it and how can we use it?
2- Provide an application scenario
3- Provide a method to apply in python (if possible)
4- Provide a follow up question about that topic
5- Schedule a chatgpt chat to send notification (Simulated)

Ensure the response is formatted in valid Markdown.
"""

    if not API_KEY:
        print("WARNING: GEMINI_API_KEY not found. Using MOCK response for testing.")
        return f"""Topic: {topic_name} (Mock)

1- Formal Definition:
Mock definition for {topic_name}...

2- Application Scenario:
Mock scenario...

3- Python Method:
```python
# Mock code
print("Hello NLP")
```

4- Follow up:
Mock question?

5- Schedule:
Standard notification...
"""
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content(prompt)
    content = response.text
    return content

def save_content(content, week_num, date_str, lesson_num):
    if not content:
        return None
        
    # Extract topic for filename (or verify it matches)
    lines = content.strip().split('\n')
    topic_line = next((line for line in lines if line.strip().startswith("Topic:")), "Topic: Unknown NLP Topic")
    topic_name = topic_line.replace("Topic:", "").strip()
    slug = slugify(topic_name)
    
    # Construct path
    # topic/nlp/week_WW/day_YYYY-MM-DD/lesson_N/
    week_dir = f"week_{week_num}"
    day_dir = f"day_{date_str}"
    lesson_dir = f"lesson_{lesson_num}"
    
    output_dir = os.path.join(TOPIC_BASE_DIR, week_dir, day_dir, lesson_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{slug}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(content)
        
    print(f"Generated: {filepath}")
    return filepath

if __name__ == "__main__":
    try:
        print("Starting content generation...")
        content = generate_content()
        if content:
            week, date_val, lesson, _ = get_lesson_info()
            filepath = save_content(content, week, date_val, lesson)
            print("Done.")
        else:
            print("No content generated (Curriculum finished or error).")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
