# AGENTS.md

This is the public portfolio repository for embodied-AI and robot-learning evidence.

This repository is the single source of truth for mature runnable code, tests, examples, and public demo projects. Keep reusable implementation work here instead of maintaining duplicate runnable code in the private learning workspace.

Only publish public evidence:

- code
- tests
- examples
- demos
- portfolio milestone records
- technical notes
- result reports
- reproducible commands
- result tables, screenshots, or videos

Do not add:

- private plans
- gap scores
- task boards
- private progress logs
- interview drafts
- resume drafts
- passwords, tokens, API keys, or credentials
- overly personal reflections
- text such as "I am weak at X" or private self-assessments

Before pushing public changes:

1. Run relevant tests. If `src/` changes, run:

```powershell
python -m unittest discover -s tests
```

2. Scan for sensitive/private content. Exclude this instructions file because it intentionally contains the scan terms:

```powershell
rg -i "password|access token|github token|api key|credential|secret|841241285|numb0118|gap analysis|task_board|progress_log|plan_review|daily_closeout|dated_schedule|0/10|weakness|private|私人|弱项" --glob "!AGENTS.md"
```

3. Keep this repository public and keep private planning content in:

```text
C:\Users\84124\Documents\New project 2\robot-learning-lab
```

Public evidence should be concise, reproducible, and portfolio-facing.
