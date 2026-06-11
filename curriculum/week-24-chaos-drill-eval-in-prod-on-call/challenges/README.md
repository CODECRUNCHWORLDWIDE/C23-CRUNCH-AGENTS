# Week 24 — Challenges

One challenge, and it is the capstone's final required deliverable: run all three chaos drills against your shipped system in a single 4-hour window, and write the postmortem.

## Index

1. **[Challenge 1 — Run the chaos drill](challenge-01-run-the-chaos-drill.md)** — in one controlled 4-hour window, run the GPU-node-loss, prompt-injection, and index-corruption drills against your Sprint B capstone, measure impact and recovery for each, and write a blameless postmortem in the standard incident format. (~3 hours)

## How to work the challenge

- This runs against your **own** Sprint B capstone — never a shared or someone else's system. One fault at a time, bounded blast radius, tested revert, you watching.
- **Plan before you inject.** Exercise 1 gave you the drill-plan structure; the challenge runs three plans. Confirm every revert works *before* the fault — especially the index restore (test it on a copy first).
- **A failing drill is a successful drill.** If the failover doesn't fire, the injection isn't stopped, or the eval doesn't catch the corruption, you found a real gap on your schedule. Write the patch, re-run, and document both the failure and the fix. The syllabus is explicit for the prompt-injection drill: *verify the defenses hold; if they do not, write the patch.*
- **The postmortem is the deliverable.** Not "I ran the drills" — a written postmortem with a measured timeline per drill, what held, what didn't, recovery times, and dated action items.

This challenge closes the loop on the entire course: you built a production agentic system, you broke it on purpose, you measured how it fails and recovers, and you wrote the document that proves it. It's the difference between a capstone that *works* and one that is *production*.
