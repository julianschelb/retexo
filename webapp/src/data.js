// The app's input: one predicted edit script per line, built from the Hugging Face datasets by
// scripts/prepare_data.py and served from public/data/.
export const DATA_FILE = "records.jsonl";

export async function loadRecords(file = DATA_FILE) {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${file}`);
  if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
  const text = await response.text();
  return text.split("\n").filter((line) => line.trim()).map((line) => JSON.parse(line));
}
