# Code-Standards

## 1. Python Code-Standards

- **PEP8 Konformität:** Alle Python-Dateien müssen den Richtlinien des [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/) entsprechen. Dies wird durch `flake8` in der CI/CD-Pipeline erzwungen.
- **Type Hints:** Alle Funktionen, Methoden und komplexen Variablen müssen mit [Type Hints](https://docs.python.org/3/library/typing.html) versehen werden, um die Lesbarkeit und Wartbarkeit des Codes zu verbessern. `mypy` wird zur statischen Typenprüfung verwendet.
- **Google Docstrings:** Alle Module, Klassen, Funktionen und Methoden müssen [Google-Style Docstrings](https://google.github.io/styleguide/pyguide.html#pyguide%3Acontents) enthalten, die den Zweck, die Argumente, Rückgabewerte und Ausnahmen detailliert beschreiben.
- **Modul-Isolation:** Module sollten lose gekoppelt und hoch kohäsiv sein. Direkte Zirkelabhängigkeiten sind zu vermeiden. Jedes Modul sollte eine klare, einzelne Verantwortlichkeit haben.

## 2. Git Workflow

- **Feature Branches:** Neue Entwicklungen müssen in separaten Feature Branches erfolgen, die von `main` abzweigen.
- **Pull Requests (PRs):** Änderungen werden über Pull Requests in den `main`-Branch integriert. PRs erfordern mindestens zwei genehmigende Reviews.
- **Conventional Commits:** Commit-Nachrichten müssen dem [Conventional Commits Standard](https://www.conventionalcommits.org/en/v1.0.0/) folgen (z.B. `feat: Add new RSI strategy`, `fix: Correct data fetching bug`).

## 3. CI/CD Pipeline-Konfiguration

- **Automatisierte Tests:** Bei jedem Push in einen Feature Branch und vor dem Merge in `main` werden Unit- und Integrationstests automatisch ausgeführt.
- **Code-Qualitäts-Checks:** `flake8`, `mypy` und `isort` werden automatisch ausgeführt, um die Code-Qualität und Formatierung sicherzustellen.
- **Coverage Threshold:** Die Testabdeckung muss mindestens 70% betragen, um die Pipeline erfolgreich zu durchlaufen.
- **Automatisches Deployment:** Nach erfolgreichem Merge in `main` und Bestehen aller Checks wird die Anwendung automatisch in die Staging-Umgebung deployed.
