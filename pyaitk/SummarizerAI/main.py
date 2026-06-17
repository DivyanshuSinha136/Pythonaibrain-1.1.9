"""
Main entrypoint for the AI Memory Summarization System.

Usage:
    python main.py                     # run with built-in demo data
    python main.py --input memory.json  # run on custom JSON file
    python main.py --query "hello"     # query the fitted model
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ─── Bootstrap ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from .summarizer import MemorySummarizer

# ─── Built-in demo memory ─────────────────────────────────────────────────────

DEMO_MEMORY = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hello! What would you like to know?",
    "bye": "Bye! Have a great day!",
    "cbse": "Hi! What can I do for you?",
    "start cmd": "Done: cmd_start",
    "start": "Wait a moment!",
    "cmd": "Wait a moment!",
    "0": "Hi! What can I do for you?",
    "Start Cmd.": "Done: cmd_start",
    "start cmd.": "Done: cmd_start",
    "tell me a funny joke": "Apparently my password needs to be capitals only so I've changed it to LONDONMADRIDROME",
    "who are u?": "I'm your AI assistant, powered by PythonAIBrain!",
    "who created u?!": "I was built using PythonAIBrain by Divyanshu Sinha!",
    "ok": "Got it!",
    "what u can do": "# Pythonaibrain\nPythonAIBrain is a versatile, plug-and-play Python package for building offline\nintelligent AI assistants.  Visit https://pypi.org/project/pythonaibrain for\nfull documentation.\n",
    "tell me a joke": "What do you call a parrot that says \"Squawk! Pieces of nine! Pieces of nine!\"? A parrot-ey error.",
    "tell me another joke": "I've been using Vim for a long time now, mainly because I can't figure out how to exit.",
    "tell a jock": "Child: Dad, why does the sun rise in the east and set in the west? Dad: Son, it's working, don't touch it.",
    "joke": "One person's error is another person's data.",
    "jock": "Hello! What would you like to know?",
    "thanks": "Any time!",
    "hey hi!": "Hello! What would you like to know?",
    "what's your name dear?": "I'm your AI assistant, powered by PythonAIBrain!",
    "oh! greate.": "Hello! How can I help you today?",
    "What is Pythonaibrain?": "['OPEN', 'What is Pythonaibrain?']",
    "What is Pythonaibrain": "['OPEN', 'What is Pythonaibrain']",
    "Pythonaibrain.": "Hey there! What can I do for you?",
    "Who created you?": "I'm powered by PythonAIBrain, an open-source AI framework.",
    "Who is your founder": "['OPEN', 'Who is your founder']",
    "name": "I'm an AI assistant here to help you.",
    "what is your age?": "I'm ageless \u2014 I exist in the digital realm!",
    "tell me a joke.": "Why do we want intelligent terminals when there are so many stupid users?",
    "another one": "Goodbye! Come back anytime.",
    "another one joke.": "To understand recursion you must first understand recursion.",
    "what is the time?": "I'll check the time for you!",
    "ok proccedd": "Got it!",
    "well done": "You're making me blush \u2014 if AIs could blush!",
    "greate": "Hey! Ready to help.",
    "yo bro": "Hey! Ready to help.",
    "brother what's your name?": "I'm your AI assistant, powered by PythonAIBrain!",
    "Greate to hear that.": "Understood!",
    "Aw! don't get blush": "[Error] HTTPSConnectionPool...",
    "what is python": "Error: 'NoneType' object is not subscriptable",
    "tell me about Python.": "['OPEN', 'tell me about Python.']",
    "ok! bye": "Goodbye! Come back anytime.",
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_config(args) -> dict:
    return {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "ae_epochs": args.epochs,
        "ae_lr": args.lr,
        "ae_batch_size": args.batch_size,
        "device": "cpu",
        "embedding": {
            "max_features": 3000,
            "ngram_range": (1, 3),
            "sublinear_tf": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="AI Memory Summarization System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to memory JSON file (uses built-in demo if omitted)")
    parser.add_argument("--output", type=str, default="memory_report.json",
                        help="Path to save JSON report")
    parser.add_argument("--query", type=str, default=None,
                        help="Query string to test after fitting")
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Load memory
    if args.input:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(raw)} patterns from {args.input}")
    else:
        raw = DEMO_MEMORY
        logger.info(f"Using built-in demo memory ({len(raw)} patterns)")

    # Build & run pipeline
    config = build_config(args)
    summarizer = MemorySummarizer(config=config)
    summarizer.fit(raw)

    # Print report to stdout
    summarizer.print_report()

    # Save JSON report
    summarizer.export_report(args.output)
    print(f"\n✅  JSON report saved to: {args.output}")

    # Optional query
    if args.query:
        print(f"\n🔍  Query: {args.query!r}")
        result = summarizer.query(args.query)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
