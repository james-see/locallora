import os
import argparse
import json
import glob
import fitz  # PyMuPDF
from pathlib import Path
import re


def clean_text(s):
    # basic cleanup
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_text(path):
    doc = fitz.open(path)
    text = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text.append(page_text)
    return clean_text("\n\n".join(text))


def chunk_text(text, max_chars=3000, overlap=200):
    # chunk by characters preserving sentences roughly (simple approach)
    start = 0
    length = len(text)
    chunks = []
    while start < length:
        end = start + max_chars
        if end >= length:
            chunks.append(text[start:length].strip())
            break
        # try to break at last newline or sentence end within overlap window
        window = text[start : end + overlap]
        split_at = None
        # prefer double newline
        idx = window.rfind("\n\n")
        if idx != -1 and idx > 0:
            split_at = start + idx
        else:
            # prefer sentence-ending punctuation within last 200 chars
            for sep in [".", "?", "!"]:
                idx = window.rfind(sep)
                if idx != -1 and idx > max_chars - 500:  # accept recent sep
                    split_at = start + idx + 1
                    break
        if not split_at:
            split_at = end
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = split_at
    return chunks


def main(args):
    pdf_folder = Path(args.pdf_folder)
    out_text = Path(args.output_text)
    out_alpaca = Path(args.output_alpaca)

    pdf_paths = sorted(glob.glob(str(pdf_folder / "*.pdf")))
    if not pdf_paths:
        print("No PDFs found in", pdf_folder)
        return

    with (
        out_text.open("w", encoding="utf-8") as f_text,
        out_alpaca.open("w", encoding="utf-8") as f_alpaca,
    ):
        for p in pdf_paths:
            title = Path(p).stem
            print("Processing:", title)
            try:
                full_text = extract_pdf_text(p)
            except Exception as e:
                print("Failed to read", p, e)
                continue
            if not full_text:
                print("No text extracted for", p)
                continue

            # optional metadata header
            header = f"Title: {title}\nSource: {p}\n\n"
            # chunk
            chunks = chunk_text(full_text, max_chars=args.max_chars, overlap=args.overlap)
            for i, c in enumerate(chunks):
                item_text = {"text": header + c, "title": title, "source_path": p, "chunk_index": i}
                f_text.write(json.dumps(item_text, ensure_ascii=False) + "\n")
                # Alpaca-style instruction: leave output empty for manual labeling or downstream pseudo-labeling
                alpaca_item = {
                    "instruction": args.instruction,
                    "input": header + c,
                    "output": "",  # leave blank for supervised labels or fill with generated summaries later
                }
                f_alpaca.write(json.dumps(alpaca_item, ensure_ascii=False) + "\n")

    print("Wrote:", out_text, out_alpaca)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDFs to JSONL (text chunks + alpaca-style)"
    )
    parser.add_argument("--pdf-folder", default="papers", help="folder with PDFs")
    parser.add_argument("--output-text", default="papers_text.jsonl", help="chunked text output")
    parser.add_argument(
        "--output-alpaca",
        default="papers_alpaca.jsonl",
        help="alpaca-style output (output left blank)",
    )
    parser.add_argument("--max-chars", type=int, default=3000, help="max chars per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="overlap chars between chunks")
    parser.add_argument(
        "--instruction",
        default="Summarize the following research paper in 5 concise bullet points.",
        help="instruction for alpaca format",
    )
    args = parser.parse_args()
    main(args)
