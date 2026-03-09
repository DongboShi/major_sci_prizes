import pandas as pd
# import wikipediaapi
from openai import OpenAI
from io import StringIO
import os

INPUT_CSV = "wiki_nobel_laureate.csv"
OUTPUT_CSV = "career_history_results.csv"
MODEL = "gpt-5.4"
wd = os.path.expanduser("~/downloads")

os.chdir(wd)

client = OpenAI()

# def wikipedia_bio_text(wikipedia_url):
#     """Extract title from URL and fetch page text via wikipediaapi."""
#     title = wikipedia_url.split("/wiki/")[-1].replace("_", " ")
#     wiki = wikipediaapi.Wikipedia(
#         language='en',
#         user_agent='MyAwesomeApp/1.0 (contact@email.com)'
#     )
#     page = wiki.page(title)
#     if not page.exists():
#         raise Exception(f"Wikipedia page for '{title}' not found!")
#     # Get the first N sections of relevant text
#     txt = page.summary + "\n"
#     # Optionally, concatenate section texts for more details
#     for section in page.sections:
#         if section.title.lower() in {"career", "biography", "life", "early life"}:
#             txt += section.text + "\n"
#     return txt.strip()

def get_career_csv(llm_prompt, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional data expert. Reply ONLY in CSV with no commentary or markdown."},
            {"role": "user", "content": llm_prompt}
        ]
    )
    csv_text = response.choices[0].message.content
    # Clean model output
    csv_text = csv_text.replace("```csv", "").replace("```", "").strip()
    # Only include rows with commas
    lines = [line for line in csv_text.splitlines() if "," in line]
    return "\n".join(lines)

df_in = pd.read_csv(INPUT_CSV)
all_rows = []
failed = []

for idx, row in df_in.iterrows():
    name = str(row['name'])
    year = str(row['year'])
    category = str(row['category'])
    wikipedia_url = row['wikipedia_url']
    try:
        # biography = wikipedia_bio_text(wikipedia_url)
        prompt = (
            "Please extract as CSV (comma-separated-value, no markdown, no commentary) the career history of the laureate at these stages: "
            "birth, basic education, college, phd program, all employees after phd graduation (including title: postdoc, assistant professor, associate, full professor, or other), "
            "working institute when finishing Nobel prize papers, and working institute when being awarded Nobel prize. "
            "Columns: year,country,institution,stage. If a field contains commas, replace them with a space."
            f"Laureate info: {name},{year},{category},{wikipedia_url}\n"
            # f"Biography:\n{biography[:4000]}"  # Truncate to avoid exceeding prompt limit
        )
        csv_result = get_career_csv(prompt)
        df_result = pd.read_csv(StringIO(csv_result))
        # print(df_result)
        if 'name' not in df_result.columns:
            df_result['name'] = name
        all_rows.append(df_result)
        # print(all_rows)
        print(f"Processed: {name}")
    except Exception as e:
        print(f"FAILED: {name} - {e}")
        failed.append({"name": name, "error": str(e)})

if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved all career history results to {OUTPUT_CSV}.")
else:
    print("No successful results to write.")

if failed:
    pd.DataFrame(failed).to_csv("failed_cases.csv", index=False)
    print(f"{len(failed)} failures saved to failed_cases.csv.")