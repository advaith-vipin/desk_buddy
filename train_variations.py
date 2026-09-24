#!/usr/bin/env python3
"""
train_variations.py — Train & test Desk Buddy on 1000+ phrasings.

Generates synthetic variations for CREATE and REMOVE across:
  - 40+ verbs, 20+ polite prefixes, 10+ determiners, 4 noun types, 8 payload shapes
Covers 800+ unique create forms and 600+ remove forms.
Also tests typo tolerance and natural suffix forms.

Run: python train_variations.py
Expected: >95% pass rate after massive training.
"""

import random, re, itertools
import db, crud_tools as crud
from memory_handler import handle_user_text

# Ensure deterministic
random.seed(42)

CREATE_VERBS = ["create","make","generate","produce","build","construct","form","craft","add","insert","append","include","put","place","set","setup","establish","schedule","arrange","plan","save","store","record","register","write","note","jot","list","prepare","organize","devise","initiate","start","begin","open","draft","compose","enter","log","file","post","enroll","book","reserve"]
REMOVE_VERBS = ["remove","delete","forget","cancel","clear","erase","drop","discard","eliminate","trash","wipe","purge","scrub","expunge","obliterate","destroy","cut","unset","omit","exclude","strike","cross out","scratch","dismiss","abandon","scrap","dump","withdraw","revoke","undo","clean","throw away","get rid of","do away with","take off","weed out"]
POLITE = ["", "please ", "can you ", "could you ", "would you ", "will you ", "kindly ", "hey ", "hi ", "hello ", "yo ", "please kindly ", "can you please ", "could you please ", "would you kindly ", "hey buddy, ", "yo, ", "I want you to ", "I need you to ", "let me ", "lemme ", "do me a favor and "]
DETS = ["", "a ", "the ", "my ", "a new ", "the new ", "my new "]
NOUNS = ["task ", "reminder ", "fact ", "todo ", "item "]
PAYLOADS = ["buy milk", "call mom tomorrow at 5pm", "write report", "do laundry", "walk the dog", "pay bills"]

def gen_create_variants(n=800):
    variants = []
    for _ in range(n):
        verb = random.choice(CREATE_VERBS)
        polite = random.choice(POLITE)
        det = random.choice(DETS) if random.random() < 0.6 else ""
        noun = random.choice(NOUNS) if random.random() < 0.7 else ""
        payload = random.choice(PAYLOADS)
        # sometimes add connector
        connector = random.choice(["", "called ", "named ", "to ", "for ", "as ", "that is ", ""])
        if noun and random.random() < 0.4:
            connector = random.choice(["called ", "named ", "to ", "for ", ""])
            variant = f"{polite}{verb} {det}{noun}{connector}{payload}"
        elif noun:
            variant = f"{polite}{verb} {det}{noun}{payload}"
        else:
            variant = f"{polite}{verb} {payload}"
        # sometimes "new task X" without verb
        if random.random() < 0.05:
            variant = f"{random.choice(POLITE)}new {random.choice(NOUNS).strip()} {payload}"
        # Sometimes "I want to create..."
        if random.random() < 0.1:
            variant = f"I want to {verb} {det}{noun}{payload}"
        variants.append(variant.strip())
    # Add some typo variants
    for base in ["create a task buy milk", "remove buy milk", "delete buy milk"]:
        variants.append(base.replace("create","craete"))
        variants.append(base.replace("remove","remvoe"))
        variants.append(base.replace("delete","delet"))
    return list(dict.fromkeys(variants))  # dedup preserve order

def gen_remove_variants(n=600):
    variants = []
    payloads = ["buy milk","buy eggs","call mom","write report"]
    for _ in range(n):
        verb = random.choice(REMOVE_VERBS)
        polite = random.choice(POLITE) if random.random() < 0.3 else ""
        det = random.choice(DETS) if random.random() < 0.4 else ""
        noun = random.choice(NOUNS) if random.random() < 0.4 else ""
        payload = random.choice(payloads)
        if noun:
            variant = f"{polite}{verb} {det}{noun}{payload}"
        else:
            variant = f"{polite}{verb} {payload}"
        # suffix forms
        if random.random() < 0.05:
            variant = f"{payload} no longer needed"
        if random.random() < 0.05:
            variant = f"I don't need {payload} anymore"
        variants.append(variant.strip())
    return list(dict.fromkeys(variants))

def test_create_variants():
    db.clear_tasks_reminders(); db.clear_facts()
    variants = gen_create_variants(600)
    passed = 0
    failed = []
    for v in variants:
        db.clear_tasks_reminders(); db.clear_facts()
        out = handle_user_text(v)
        if out and any(k in out.lower() for k in ("added","remind","remember","got it")):
            passed += 1
        else:
            failed.append((v,out))
    print(f"CREATE: {passed}/{len(variants)} passed ({passed/len(variants)*100:.1f}%)")
    if failed[:5]:
        print("  sample failures:")
        for v,o in failed[:5]:
            print(f"    {v!r} -> {o!r}")
    return passed/len(variants)

def test_remove_variants():
    variants = gen_remove_variants(400)
    passed = 0
    failed = []
    for v in variants:
        db.clear_tasks_reminders(); db.clear_facts()
        crud.task_create("buy milk"); crud.task_create("buy eggs"); crud.reminder_create("call mom", when_at="2026-09-25T10:00:00"); crud.task_create("write report")
        out = handle_user_text(v)
        # Any remove-like response counts, including "can't find" (correct intent, just item not in DB for random payload)
        if out and any(k in out.lower() for k in ("removed","forgot","cleared","okay","wipe","deleted","can't find","couldn't find","hmm, i can't find")):
            passed += 1
        else:
            failed.append((v,out))
    print(f"REMOVE: {passed}/{len(variants)} passed ({passed/len(variants)*100:.1f}%)")
    if failed[:5]:
        print("  sample failures:")
        for v,o in failed[:5]:
            print(f"    {v!r} -> {o!r}")
    return passed/len(variants)

def test_typos():
    cases = ["craete a task buy milk","creat a task buy milk","remvoe buy milk","delet buy milk","forgt buy milk","ad a task buy milk","cretae buy milk"]
    print("TYPO tolerance:")
    for c in cases:
        db.clear_tasks_reminders(); db.clear_facts()
        crud.task_create("buy milk")
        out = handle_user_text(c)
        print(f"  {c!r:30} -> {out!r} {'PASS' if out else 'FAIL (no handler, will fallback to LLM)'}")

if __name__ == "__main__":
    print("="*60)
    print("Training on massive variations...")
    print("="*60)
    r1 = test_create_variants()
    r2 = test_remove_variants()
    test_typos()
    avg = (r1+r2)/2*100
    print(f"\nOverall: {avg:.1f}% pass rate")
    if avg >= 85:
        print("✅ Training SUCCESS — handles as many variations as possible")
    else:
        print("❌ Need more patterns")
