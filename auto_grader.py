'''
Main script for testing the 2026 assignment.
Runs the tests on the results json file with binary grading (0/1).
'''

import argparse
import json
import re

def get_args():
    parser = argparse.ArgumentParser(description='Language Modeling 2026')
    parser.add_argument('test', type=str, help='The test to perform.')
    return parser.parse_args()

def test_link(results):
    # Read link
    link = results.get("link")
    if not link:
        return "link field not found in results.json"
    
    # Make sure link contains usp=sharing
    if "usp=sharing" not in link:
        return f"Link {link} doesn't seem to have share access"
    return 1
    

def test_preprocess(results):
    expected = 1804
    if results.get("vocab_length") == expected:
        return 1
    return f"Vocab length is {results.get('vocab_length')}, expected {expected}"

def test_build_lm(results):
    checks = {
        "english_2_gram_length": 748,
        "english_3_gram_length": 8239,
        "french_3_gram_length": 8286,
        "spanish_3_gram_length": 8469
    }
    
    for key, expected in checks.items():
        if results.get(key) != expected:
            return f"{key}: expected {expected}, got {results.get(key)}"
    return 1
    
def test_eval(results):
    try:
        en_on_en = float(results["en_on_en"])
        en_on_fr = float(results["en_on_fr"])
        en_on_tl = float(results["en_on_tl"])
        en_on_nl = float(results["en_on_nl"])
    except (KeyError, ValueError, TypeError):
        return "Missing or invalid perplexity values"

    if not (en_on_en < en_on_fr < min(en_on_tl, en_on_nl)):
        return "English model should perform best on English text, then French, then others"
    
    if not (en_on_en < en_on_fr < max(en_on_tl, en_on_nl)):
        return "Expected increasing perplexity trend"
        
    return 1

def test_generate(results):
    if not results.get("english_2_gram", "").startswith("I am"):
        return "English 2-gram failure"
    if not results.get("english_3_gram", "").startswith("I am"):
        return "English 3-gram failure"
    if not results.get("french_3_gram", "").startswith("Je suis"):
        return "French 3-gram failure"
    
    return 1

def main():
    args = get_args()
    try:
        with open('results.json', 'r') as f:
            full_results = json.load(f)
    except Exception as e:
        print(f"Error reading results.json: {e}")
        return

    result = "Invalid test"
    
    if args.test == 'test_link':
        result = test_link(full_results)
    elif args.test == 'test_preprocess':
        result = test_preprocess(full_results.get("test_preprocess", {}))
    elif args.test == 'test_build_lm':
        result = test_build_lm(full_results.get("test_build_lm", {}))
    elif args.test == 'test_eval':
        result = test_eval(full_results.get("test_eval", {}))
    elif args.test == 'test_generate':
        result = test_generate(full_results.get("test_generate", {}))

    # Print result for autograder
    print(result)

if __name__ == '__main__':
    main()
